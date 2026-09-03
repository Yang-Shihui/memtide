"""Multimodal ingestion: turn image/audio/file message parts into text + attachments.

Strategy (Mem0-style, the industry norm): media content is normalized to text
at write time — images go through a vision model, audio through an optional
STT endpoint — and the resulting description flows into the normal text
extraction pipeline. The original bytes are kept as a content-addressed
attachment (sha256) so retrieval can hand the asset back to the agent.

Zero new dependencies: urllib + base64 only.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import urllib.request
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

_IMAGE_PROMPT = (
    "Describe this image for a long-term memory system. State the concrete "
    "facts it shows (people, objects, places, visible text, numbers) in 1-3 "
    "short sentences. No speculation, no filler."
)

_MIME_BY_EXT = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml",
    ".bmp": "image/bmp",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
    ".ogg": "audio/ogg", ".flac": "audio/flac",
    ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
    ".pdf": "application/pdf", ".txt": "text/plain", ".md": "text/markdown",
}


@dataclass
class Attachment:
    """A media asset referenced by one or more memories."""

    id: str                 # sha256 prefix (16 hex chars)
    kind: str               # image | audio | video | file
    source: str             # data-URL / https URL / media file name
    mime: str = ""
    sha256: str = ""
    description: Optional[str] = None  # caption / transcript / vision output

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "source": self.source,
                "mime": self.mime, "sha256": self.sha256,
                "description": self.description}


# ---------------------------------------------------------------- storage ----

def save_media(data: bytes, media_dir: str, mime: str = "") -> Tuple[str, str, str]:
    """Content-addressed store: write bytes as ``<sha256><ext>`` under media_dir.

    Returns ``(sha256, file_name, mime)``. Identical bytes land on the same
    file, so re-sending an image is a natural dedup.
    """
    import hashlib

    sha = hashlib.sha256(data).hexdigest()
    ext = ""
    if mime:
        ext = mimetypes.guess_extension(mime) or ""
        if ext == ".jpe":  # mimetypes' odd choice for image/jpeg
            ext = ".jpg"
    if not ext:
        ext = ".bin"
    os.makedirs(media_dir, exist_ok=True)
    name = sha + ext
    path = os.path.join(media_dir, name)
    if not os.path.exists(path):  # same bytes -> same name, write once
        tmp = path + f".tmp{uuid.uuid4().hex[:8]}"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    return sha, name, (mime or _MIME_BY_EXT.get(ext, "application/octet-stream"))


def media_path(media_dir: str, sha256: str) -> Optional[str]:
    """Locate a stored asset by sha256 (any extension)."""
    if not re.fullmatch(r"[0-9a-f]{64}", sha256 or ""):
        return None
    if not os.path.isdir(media_dir):
        return None
    for name in os.listdir(media_dir):
        if name.split(".")[0] == sha256:
            return os.path.join(media_dir, name)
    return None


# ---------------------------------------------------------------- fetching ---

def _data_url_parts(url: str) -> Tuple[str, bytes]:
    head, _, b64 = url.partition(",")
    m = re.match(r"data:([^;]+)(?:;base64)?", head)
    mime = m.group(1) if m else "application/octet-stream"
    return mime, base64.b64decode(b64)


def _guard_url(url: str) -> None:
    """Block SSRF: server-side fetch must never reach loopback/link-local/
    private networks or cloud metadata endpoints."""
    import ipaddress
    from urllib.parse import urlparse

    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if host in ("localhost", "metadata.google.internal"):
        raise ValueError(f"media host blocked: {host}")
    if host.endswith(".local") or host.endswith(".internal"):
        raise ValueError(f"media host blocked: {host}")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return  # DNS name: literal-IP private ranges already covered below
    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved \
            or ip.is_multicast or ip.is_unspecified:
        raise ValueError(f"media IP blocked: {host}")


def _fetch_url(url: str, cap: int) -> Tuple[str, bytes]:
    _guard_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": "memtide/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        mime = resp.headers.get("Content-Type", "").split(";")[0].strip()
        data = resp.read(cap + 1)
    if len(data) > cap:
        raise ValueError(f"media at {url} exceeds {cap} bytes")
    return mime, data


def _read_path(path: str, cap: int) -> Tuple[str, bytes]:
    try:
        size = os.path.getsize(path)
        if size > cap:
            raise ValueError(f"media file {path} exceeds {cap} bytes")
        mime = mimetypes.guess_type(path)[0] or ""
        with open(path, "rb") as f:
            return mime, f.read()
    except OSError as e:
        raise ValueError(f"cannot read media file {path!r}: {e}") from e


def _resolve_media(part: Dict[str, Any], cfg) -> Tuple[str, bytes, str]:
    """(mime, bytes, source) from one message part."""
    ref = ""
    if isinstance(part.get("image_url"), dict):
        ref = part["image_url"].get("url", "")
    elif isinstance(part.get("input_audio"), dict):
        audio = part["input_audio"]
        data = base64.b64decode(audio.get("data", ""))
        return f"audio/{audio.get('format', 'wav')}", data, "inline"
    for key in ("url", "path", "data"):
        if isinstance(part.get(key), str) and part[key]:
            ref = part[key]
            break
    if not ref:
        raise ValueError("media part has no url/path/data")
    if ref.startswith("data:"):
        mime, data = _data_url_parts(ref)
        return mime, data, ref
    if ref.startswith("http://") or ref.startswith("https://"):
        mime, data = _fetch_url(ref, cfg.max_media_bytes)
        return mime, data, ref
    if not getattr(cfg, "media_allow_paths", False):
        # local-path ingest is opt-in: an internet-facing REST server must not
        # let callers turn {"path": "/etc/passwd"} into a downloadable asset
        raise ValueError(
            "local media paths are disabled (media_allow_paths=False); "
            "pass a data: or https:// URL instead")
    mime, data = _read_path(ref, cfg.max_media_bytes)  # local file path
    return mime, data, os.path.basename(ref)


# ------------------------------------------------------------ describing -----

def _vision_post(base_url: str, api_key: str, payload: dict) -> Optional[dict]:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode())
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError):
        return None  # graceful: no vision available -> attachment without caption


def describe_image(data_url: str, cfg) -> Optional[str]:
    """Caption an image via any OpenAI-compatible vision endpoint.

    Returns None (silently, media is still stored) when no endpoint is
    reachable or the configured model rejects images.
    """
    base = cfg.vision_base_url or cfg.llm_base_url
    model = cfg.vision_model or cfg.llm_model
    api_key = cfg.vision_api_key or cfg.resolve_api_key()
    if not api_key:
        return None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": _IMAGE_PROMPT},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]}],
        "temperature": 0.0,
        "max_tokens": 300,
    }
    data = _vision_post(base, api_key, payload)
    if not data:
        return None
    try:
        return (data["choices"][0]["message"].get("content") or "").strip() or None
    except (KeyError, IndexError, TypeError):
        return None


def transcribe_audio(data: bytes, filename: str, mime: str, cfg) -> Optional[str]:
    """Optional STT via an OpenAI-compatible /audio/transcriptions endpoint."""
    base = cfg.stt_base_url or cfg.llm_base_url
    model = cfg.stt_model or ""
    api_key = cfg.vision_api_key or cfg.resolve_api_key()
    if not (model and api_key):
        return None
    boundary = uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'.encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{filename}"\r\nContent-Type: {mime or "application/octet-stream"}\r\n\r\n'.encode(),
        data, f"\r\n--{boundary}--\r\n".encode(),
    ]
    req = urllib.request.Request(
        f"{base.rstrip('/')}/audio/transcriptions", data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return (json.loads(resp.read().decode()).get("text") or "").strip() or None
    except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError):
        return None


# ---------------------------------------------------------- message parts ----

def _kind_of(part: Dict[str, Any]) -> Optional[str]:
    t = str(part.get("type", "")).lower()
    if t in ("image_url", "image", "input_image"):
        return "image"
    if t in ("input_audio", "audio"):
        return "audio"
    if t in ("video", "input_video"):
        return "video"
    if t in ("file", "input_file", "media", "document"):
        return "file"
    return None


def process_part(part: Dict[str, Any], cfg) -> Optional[Attachment]:
    """One media message part -> stored Attachment (with description if we can)."""
    kind = _kind_of(part)
    if kind is None:
        return None
    mime, data, source = _resolve_media(part, cfg)
    if len(data) > cfg.max_media_bytes:
        raise ValueError(f"media exceeds {cfg.max_media_bytes} bytes")
    sha, name, mime = save_media(data, cfg.media_dir, mime)
    att = Attachment(id=sha[:16], kind=kind, source=name, mime=mime, sha256=sha,
                     description=part.get("caption") or None)
    if att.description is None and cfg.multimodal_enabled:
        if kind == "image":
            data_url = source if source.startswith("data:") else (
                f"data:{mime};base64," + base64.b64encode(data).decode())
            att.description = describe_image(data_url, cfg)
        elif kind == "audio":
            att.description = transcribe_audio(data, name, mime, cfg)
    return att


def process_messages(messages: Any, cfg) -> Tuple[str, List[Dict[str, Any]]]:
    """Flatten messages (string | [{'role','content'}]) to a transcript.

    Media parts become ``[image: <description>]`` placeholders in the
    transcript AND Attachment dicts on the side. Raises ValueError for
    unresolvable/oversized media so callers can map it to a 400.
    """
    if isinstance(messages, str):
        return messages, []
    lines: List[str] = []
    attachments: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role", "user") if isinstance(m, dict) else getattr(m, "role", "user")
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        if content is None:  # tool-call messages etc. carry no content
            continue
        if isinstance(content, str):
            lines.append(f"{role}: {content}")
            continue
        if not isinstance(content, list):
            lines.append(f"{role}: {content}")
            continue
        texts: List[str] = []
        for part in content:
            if not isinstance(part, dict) or _kind_of(part) is None:
                texts.append(str(part.get("text", part)) if isinstance(part, dict) else str(part))
                continue
            if not cfg.multimodal_enabled:
                texts.append(f"[{part.get('type')} ignored]")
                continue
            att = process_part(part, cfg)
            if att is None:
                continue
            attachments.append(att.to_dict())
            label = f"[{att.kind}"
            if att.description:
                label += f": {att.description}"
            texts.append(label + "]")
        lines.append(f"{role}: " + " ".join(t for t in texts if t))
    return "\n".join(lines), attachments

"""REST API for Memtide (stdlib http.server, threaded; the only runtime
dependency is psycopg).

Run:  python -m memtide serve --port 8300   (MEMTIDE_PG_DSN from env)

Endpoints (all JSON):
    POST   /memories           add     {"text"|"messages", "user_id", "agent_id", "run_id", "infer", "metadata"}
    GET    /memories           list    ?user_id=&agent_id=&run_id=&limit=
    GET    /memories/{id}      get
    PUT    /memories/{id}      update  {"text": "..."}
    DELETE /memories/{id}      delete  ?hard=true
    POST   /search             search  {"query", "user_id", "limit", "include_forgotten"}
    POST   /consolidate        reflection pass {"user_id", "agent_id", "run_id"}
    POST   /rebuild            rebuild the vector index from PostgreSQL
    GET    /context            render  ?user_id=&query=
    GET    /history            audit   ?memory_id=&limit=
    GET    /export             JSONL dump ?user_id=&embeddings=&include_invalid=&download
    POST   /import             {"lines": [...], "on_conflict": "skip|overwrite"}
    POST   /compact            near-duplicate compaction {"user_id", "threshold"?}
    POST   /media/gc           orphan media cleanup {"delete": true}
    GET    /media/{sha256}     fetch a stored media asset (images, audio, files)
    GET    /stats
    POST   /reset              wipe (requires body {"confirm": "RESET"})

The landing page is served at the ROOT (http://host:port/); the management
console lives at /console (legacy /ui/*), and /docs & /api serve the API
docs. When MEMTIDE_API_KEY is set, every API endpoint above
requires the key via "X-API-Key" or "Authorization: Bearer".

A single engine is shared across request threads behind the ENGINE's lock
(all routes hold it, so REST handlers, background adds and the
auto-reflect loop are mutually exclusive).
"""

from __future__ import annotations

import hmac
import json
import os
import mimetypes
import threading
from pathlib import Path
from typing import Optional
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .config import MemoryConfig
from .engine import MemoryEngine

# paths that speak the JSON API (gated when an API key is configured);
# everything else serves the static console (always public)
API_PREFIXES = ("/memories", "/search", "/context", "/history", "/stats",
                "/consolidate", "/reset", "/rebuild", "/compact",
                "/media", "/export", "/import")


class _BodyTooLarge(Exception):
    pass


def _safe_error(e: Exception) -> dict:
    """Desensitised 500 body: exception type only, message truncated to 200
    chars so PG DSNs / URLs / paths never leak to API callers."""
    msg = str(e)[:200]
    return {"error": f"{type(e).__name__}: {msg}" if msg else type(e).__name__}


class _State:
    engine: MemoryEngine
    lock: threading.RLock


def _static_dir() -> Path:
    env = os.environ.get("MEMTIDE_STATIC_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "static"


def make_handler(engine: MemoryEngine, api_key: Optional[str] = None) -> type:
    state = _State()
    state.engine = engine
    # One lock for everything: the engine's own RLock. A separate server lock
    # would leave blind spots (background adds vs REST writes, direct store
    # reads vs engine writes), so every route holds the same lock.
    state.lock = engine._lock
    state.api_key = api_key

    class Handler(BaseHTTPRequestHandler):
        timeout = 60  # drop slow/stalled clients instead of pinning a thread

        def log_message(self, *a):  # quiet by default
            pass

        # ---- helpers -----------------------------------------------------
        def _authed(self) -> bool:
            """API-key gate (state.api_key unset = open, dev default).
            Only API paths are gated; the static console is reachable so it
            can load and ask for the key in the browser."""
            if not state.api_key:
                return True
            p = urlparse(self.path).path
            if not p.startswith(API_PREFIXES):
                return True
            got = self.headers.get("X-API-Key")
            if not got:
                auth = self.headers.get("Authorization", "")
                got = auth[7:] if auth.startswith("Bearer ") else ""
            return hmac.compare_digest(got, state.api_key)

        def _deny(self) -> None:
            self._json(401, {"error": "missing or invalid API key "
                                      "(X-API-Key header or Authorization: Bearer)"})

        def _send(self, code: int, headers: dict, body: bytes) -> None:
            """Send a full response, swallowing client-abort noise (browser
            closing a tab / cancelling a poll mid-write raises
            BrokenPipeError — log spam, not a server fault)."""
            try:
                self.send_response(code)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.end_headers()
                self._write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _write(self, body: bytes) -> None:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _json(self, code: int, payload) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode()
            self._send(code, {"Content-Type": "application/json; charset=utf-8",
                              "Content-Length": str(len(body))}, body)

        def _body(self) -> dict:
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return {}
            if n <= 0:
                return {}
            if n > 32 * 1024 * 1024:  # hard cap: no unbounded reads
                raise _BodyTooLarge()
            try:
                data = json.loads(self.rfile.read(n).decode())
                return data if isinstance(data, dict) else {}
            except (ValueError, UnicodeDecodeError):
                return {}

        def _qs(self) -> dict:
            # keep_blank_values: a bare ?download must survive as "" so the
            # documented flag form works
            return {k: v[0] for k, v in
                    parse_qs(urlparse(self.path).query, keep_blank_values=True).items()}

        def _scoped(self, params: dict) -> dict:
            return {k: params.get(k) or None for k in ("user_id", "agent_id", "run_id")}

        @staticmethod
        def _bool(value, default: bool) -> bool:
            """JSON booleans pass through; the string 'false'/'0'/'no' must
            not be silently truthy."""
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() not in ("", "false", "0", "no", "off")
            return bool(value)

        def _flag(self, params: dict, name: str) -> bool:
            """Bare query flag (?name) is true; ?name=false/0/no is false."""
            if name not in params:
                return False
            v = (params[name] or "").strip().lower()
            if not v:
                return True  # bare flag: ?download means download
            return v not in ("false", "0", "no", "off")

        @staticmethod
        def _int(value, default: int, lo: int = 1, hi: int = 500) -> int:
            try:
                return max(lo, min(hi, int(value)))
            except (TypeError, ValueError):
                return default

        @staticmethod
        def _float(value, default: Optional[float], lo: float = 0.0, hi: float = 1.0) -> Optional[float]:
            if value is None:
                return default
            try:
                v = float(value)
            except (TypeError, ValueError):
                return None  # signals 400 to the caller
            if not (lo <= v <= hi) or v != v:  # NaN fails the range check
                return None
            return v

        # ---- routes -------------------------------------------------------
        def do_POST(self):
            if not self._authed():
                return self._deny()
            try:
                self._route_do_POST()
            except _BodyTooLarge:
                self._json(413, {"error": "request body too large (max 32MB)"})
            except Exception as e:  # malformed params, backend down, ...
                self._json(500, _safe_error(e))

        def _route_do_POST(self):
            path = urlparse(self.path).path.rstrip("/") or "/"
            body = self._body()
            if path == "/memories":
                text = body.get("text")
                messages = body.get("messages")
                if not text and not messages:
                    return self._json(400, {"error": "need 'text' or 'messages'"})
                if body.get("metadata") is not None and not isinstance(body["metadata"], dict):
                    return self._json(400, {"error": "'metadata' must be an object"})
                ts = body.get("timestamp")
                try:
                    with state.lock:
                        res = state.engine.add(
                            messages if messages else text,
                            user_id=body.get("user_id") or "default",
                            agent_id=body.get("agent_id"),
                            run_id=body.get("run_id"),
                            metadata=body.get("metadata"),
                            infer=self._bool(body.get("infer"), True),
                            timestamp=ts,
                        )
                except ValueError as e:
                    return self._json(400, {"error": str(e)})
                return self._json(201, res.to_dict())
            if path == "/search":
                query = body.get("query")
                if not query:
                    return self._json(400, {"error": "need 'query'"})
                with state.lock:
                    hits = state.engine.search(
                        query,
                        user_id=body.get("user_id") or "default",
                        agent_id=body.get("agent_id"),
                        run_id=body.get("run_id"),
                        limit=self._int(body.get("limit"), 10),
                        include_forgotten=self._bool(body.get("include_forgotten"), False),
                        memory_type=body.get("memory_type") or None,
                        slot=body.get("slot") or None,
                    )
                return self._json(200, [h.to_dict() for h in hits])
            if path == "/consolidate":
                with state.lock:
                    report = state.engine.consolidate_background(
                        user_id=body.get("user_id") or "default",
                        agent_id=body.get("agent_id"),
                        run_id=body.get("run_id"),
                    )
                return self._json(200, report)
            if path == "/rebuild":
                with state.lock:
                    n = state.engine.rebuild_index()
                return self._json(200, {"reindexed": n})
            if path == "/reset":
                if body.get("confirm") != "RESET":
                    return self._json(400, {"error": 'reset requires body {"confirm": "RESET"}'})
                with state.lock:
                    state.engine.reset()
                return self._json(200, {"ok": True})
            if path == "/compact":
                threshold = self._float(body.get("threshold"), None)
                if body.get("threshold") is not None and threshold is None:
                    return self._json(400, {"error": "threshold must be a number in [0, 1]"})
                with state.lock:
                    report = state.engine.compact(
                        user_id=body.get("user_id") or "default",
                        agent_id=body.get("agent_id"),
                        run_id=body.get("run_id"),
                        threshold=threshold,
                    )
                return self._json(200, report)
            if path == "/media/gc":
                with state.lock:
                    report = state.engine.media_gc(delete=self._bool(body.get("delete"), False))
                return self._json(200, report)
            if path == "/import":
                lines = body.get("lines")
                if not isinstance(lines, list):
                    return self._json(400, {"error": "need 'lines': [ {...}, ... ] "
                                                     "(JSONL objects as produced by /export)"})
                if body.get("on_conflict") not in (None, "skip", "overwrite"):
                    return self._json(400, {"error": "on_conflict must be skip|overwrite"})
                with state.lock:
                    stats = state.engine.import_jsonl(
                        lines, on_conflict=body.get("on_conflict") or "skip")
                return self._json(200, stats)
            return self._json(404, {"error": f"unknown path {path}"})

        def do_GET(self):
            if not self._authed():
                return self._deny()
            try:
                self._route_do_GET()
            except Exception as e:  # malformed params, backend down, ...
                self._json(500, _safe_error(e))

        def _serve_static(self, path: str) -> None:
            """Static hosting. ``/`` is the landing page (falls back to the
            console when no landing.html exists); ``/console`` (legacy
            ``/ui``) hosts the management SPA with fallback to its shell."""
            root = _static_dir().resolve()
            rel = path.lstrip("/")
            is_console = rel.startswith("console") or rel.startswith("ui")
            for pfx in ("console/", "ui/"):
                if rel.startswith(pfx):
                    rel = rel[len(pfx):]
                    break
            if rel in ("", "index.html") and not is_console:
                landing = root / "landing.html"
                file = landing if landing.is_file() else root / "index.html"
            else:
                if rel in ("docs", "api"):  # extensionless page routes
                    rel = "api-docs.html"
                file = (root / (rel or "index.html")).resolve()
                if root not in file.parents and file != root:
                    return self._json(404, {"error": "not found"})
                if not file.is_file():
                    # SPA fallback inside the console; landing fallback outside
                    file = root / ("index.html" if is_console else "landing.html")
                    if not file.is_file():
                        file = root / "index.html"
                    if not file.is_file():
                        return self._json(404, {"error": "UI not built; run webui build"})
            ctype = mimetypes.guess_type(str(file))[0] or "application/octet-stream"
            body = file.read_bytes()
            self._send(200, {"Content-Type": ctype,
                             "Content-Length": str(len(body))}, body)

        def _serve_media(self, name: str) -> None:
            """Serve a content-addressed media asset by sha256 (immutable)."""
            from . import multimodal

            sha = name.split(".")[0].lower()
            file_path = multimodal.media_path(state.engine.cfg.media_dir, sha)
            if not file_path or not os.path.isfile(file_path):
                return self._json(404, {"error": "media not found"})
            ctype = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            with open(file_path, "rb") as f:
                body = f.read()
            self._send(200, {
                "Content-Type": ctype,
                "Content-Length": str(len(body)),
                "Cache-Control": "public, max-age=31536000, immutable",
                # media bytes are attacker-influenced via uploads: never let
                # the browser sniff or render them in-site (the console shows
                # them through blob URLs, which ignore this header)
                "X-Content-Type-Options": "nosniff",
                "Content-Disposition": f'attachment; filename="{name}"',
            }, body)

        def _route_do_GET(self):
            url = urlparse(self.path)
            path = url.path.rstrip("/") or "/"
            params = {k: v[0] for k, v in
                      parse_qs(url.query, keep_blank_values=True).items()}
            # the console IS the root; /ui/* kept as a legacy alias.
            # non-API paths fall through to the SPA at the end of this method
            if path.startswith("/media/"):
                return self._serve_media(path.split("/")[-1])
            if path == "/memories":
                scope = self._scoped(params)
                include_invalid = self._flag(params, "include_invalid")
                with state.lock:
                    if include_invalid:
                        mems = state.engine.store.all_rows(
                            user_id=scope["user_id"], agent_id=scope["agent_id"],
                            run_id=scope["run_id"], include_invalid=True,
                            limit=self._int(params.get("limit"), 100))
                    else:
                        mems = state.engine.get_all(
                            user_id=scope["user_id"], agent_id=scope["agent_id"],
                            run_id=scope["run_id"], limit=self._int(params.get("limit"), 100))
                return self._json(200, [m.to_dict() for m in mems])
            if path.startswith("/memories/"):
                mid = path.split("/")[-1]
                include_invalid = self._flag(params, "include_invalid")
                with state.lock:
                    mem = state.engine.get(mid)
                # soft-deleted (invalidated) memories stay in the DB for audit,
                # but the API reports them as gone unless explicitly included
                if mem and (not mem.invalid_at or include_invalid):
                    return self._json(200, mem.to_dict())
                return self._json(404, {"error": "not found"})
            if path == "/context":
                with state.lock:
                    block = state.engine.render_context(
                        user_id=params.get("user_id") or "default",
                        query=params.get("query"))
                return self._json(200, {"context": block})
            if path == "/history":
                with state.lock:
                    hist = state.engine.get_history(
                        memory_id=params.get("memory_id"),
                        limit=self._int(params.get("limit"), 100))
                return self._json(200, hist)
            if path == "/export":
                with state.lock:
                    lines = state.engine.export_jsonl(
                        user_id=params.get("user_id"),
                        agent_id=params.get("agent_id"),
                        run_id=params.get("run_id"),
                        include_invalid=self._bool(params.get("include_invalid"), True)
                        if params.get("include_invalid") else True,
                        include_embeddings=self._bool(params.get("embeddings"), True)
                        if params.get("embeddings") else True,
                    )
                body = ("\n".join(lines) + ("\n" if lines else "")).encode()
                headers = {"Content-Type": "application/x-ndjson; charset=utf-8",
                           "Content-Length": str(len(body))}
                if self._flag(params, "download"):
                    headers["Content-Disposition"] = 'attachment; filename="memtide-export.jsonl"'
                self._send(200, headers, body)
                return
            if path == "/stats":
                with state.lock:
                    return self._json(200, state.engine.stats())
            if path.startswith(API_PREFIXES):
                return self._json(404, {"error": f"unknown path {path}"})
            return self._serve_static(path)

        def do_PUT(self):
            if not self._authed():
                return self._deny()
            try:
                self._route_do_PUT()
            except Exception as e:  # malformed params, backend down, ...
                self._json(500, _safe_error(e))

        def _route_do_PUT(self):
            path = urlparse(self.path).path.rstrip("/")
            if not path.startswith("/memories/"):
                return self._json(404, {"error": "unknown path"})
            mid = path.split("/")[-1]
            text = self._body().get("text")
            if not text:
                return self._json(400, {"error": "need 'text'"})
            with state.lock:
                ok = state.engine.update(mid, text)
            return self._json(200, {"updated": ok}) if ok else self._json(404, {"error": "not found"})

        def do_DELETE(self):
            if not self._authed():
                return self._deny()
            try:
                self._route_do_DELETE()
            except Exception as e:  # malformed params, backend down, ...
                self._json(500, _safe_error(e))

        def _route_do_DELETE(self):
            path = urlparse(self.path).path.rstrip("/")
            if not path.startswith("/memories/"):
                return self._json(404, {"error": "unknown path"})
            mid = path.split("/")[-1]
            hard = self._qs().get("hard", "").lower() == "true"
            with state.lock:
                ok = state.engine.delete(mid, hard=hard)
            return self._json(200, {"deleted": ok}) if ok else self._json(404, {"error": "not found"})

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8300,
          config: MemoryConfig | None = None) -> ThreadingHTTPServer:
    from .config import config_from_env

    cfg = config or config_from_env()
    engine = MemoryEngine(cfg)
    if cfg.auto_reflect_seconds > 0:
        engine.enable_auto_reflect(cfg.auto_reflect_seconds)
    handler = make_handler(engine, api_key=cfg.api_key)
    httpd = ThreadingHTTPServer((host, port), handler)
    return httpd


def serve_forever(host: str = "127.0.0.1", port: int = 8300) -> None:
    # deployment via env vars (Docker): MEMTIDE_PG_DSN/QDRANT_URL/LLM_*/DASHSCOPE_*
    from .config import config_from_env

    cfg = config_from_env()
    engine = MemoryEngine(cfg)
    backend = f"storage={cfg.storage_backend} vector={cfg.vector_backend} llm={cfg.llm_backend}"
    if cfg.auto_reflect_seconds > 0:
        engine.enable_auto_reflect(cfg.auto_reflect_seconds)
    auth = "api-key=ON" if cfg.api_key else "open"
    handler = make_handler(engine, api_key=cfg.api_key)
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"memtide REST API on http://{host}:{port}  ({backend}, {auth})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()

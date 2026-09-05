"""Embedding backends: any OpenAI-compatible /embeddings endpoint.

Swap in ``OpenAIEmbedder`` (or any OpenAI-compatible /embeddings endpoint)
for real semantic quality; retrieval code is backend-agnostic.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.request
from typing import List, Optional


def _hash_bucket(token: str, dim: int, salt: int = 0) -> int:
    h = hashlib.blake2b(token.encode("utf-8"), digest_size=8, key=str(salt).encode()).digest()
    return int.from_bytes(h, "big") % dim


class OpenAIEmbedder:
    """Embeddings via any OpenAI-compatible POST {base_url}/embeddings.

    Works with OpenAI itself and with DashScope's compatible-mode endpoint
    (that is how ``DashScopeEmbedder`` is implemented — it is this class with
    a different default base_url and key resolver).
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._cache: dict = {}
        self.dim: Optional[int] = None  # probed on first embed

    def embed(self, text: str) -> List[float]:
        if text in self._cache:
            return self._cache[text]
        vecs = self.embed_batch([text])
        return vecs[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """One API call for the not-yet-cached texts (OpenAI/DashScope accept
        list input); falls back to per-text calls if the endpoint balks."""
        out: List[List[float]] = []
        missing: List[str] = []
        for t in texts:
            if t in self._cache:
                out.append(self._cache[t])
            else:
                missing.append(t)
        if missing:
            got = self._request_batch(missing)
            for t, v in zip(missing, got):
                self.dim = len(v)
                if len(self._cache) > 2048:
                    self._cache.clear()
                self._cache[t] = v
                out.append(v)
        return out

    def _request_batch(self, texts: List[str]) -> List[List[float]]:
        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({"input": texts, "model": self.model}).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode())
        rows = sorted(data["data"], key=lambda r: r.get("index", 0))
        return [r["embedding"] for r in rows]


class DashScopeEmbedder(OpenAIEmbedder):
    """Alibaba DashScope text embeddings (qwen3.7-text-embedding etc.) via the
    OpenAI-compatible endpoint."""

    def __init__(self, api_key: str, model: str = "qwen3.7-text-embedding",
                 base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
                 timeout: float = 30.0):
        super().__init__(base_url, api_key, model, timeout)


class CachedEmbedder:
    """Wrapper adding an exact-text -> vector cache over any embedder.

    Repeated texts (dedup checks, re-embeds of unchanged facts, gate scans)
    stop hitting the API entirely. ``base_embedder()`` unwraps it so isinstance
    checks on the underlying backend keep working."""

    def __init__(self, inner, max_entries: int = 4096):
        self.inner = inner
        self.dim = getattr(inner, "dim", None)
        self._cache: dict = {}
        self._max = max_entries

    def embed(self, text: str) -> List[float]:
        vec = self._cache.get(text)
        if vec is None:
            vec = self.inner.embed(text)
            if len(self._cache) >= self._max:
                self._cache.clear()
            self._cache[text] = vec
            self.dim = getattr(self.inner, "dim", None) or self.dim
        return vec

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            got = self.inner.embed_batch(missing)
            for t, v in zip(missing, got):
                if len(self._cache) >= self._max:
                    self._cache.clear()
                self._cache[t] = v
        self.dim = getattr(self.inner, "dim", None) or self.dim
        return [self._cache[t] for t in texts]


def base_embedder(embedder):
    """Unwrap CachedEmbedder (and any similar wrapper) down to the backend."""
    while hasattr(embedder, "inner"):
        embedder = embedder.inner
    return embedder


def make_embedder(config) -> object:
    """Backend factory honoring config.embedding_backend.

    'auto': dashscope (if DASHSCOPE_API_KEY) -> openai (if OPENAI/LLM_API_KEY).
    """
    backend = config.embedding_backend
    if backend not in ("auto", "openai", "dashscope"):
        raise ValueError(
            f"embedding_backend={backend!r} 不支持：请使用 auto/openai/dashscope")
    if backend == "dashscope":
        api_key = config.resolve_dashscope_key()
        if not api_key:
            raise ValueError("embedding_backend=dashscope 需要 DASHSCOPE_API_KEY")
        return DashScopeEmbedder(api_key, config.dashscope_embedding_model,
                                 config.dashscope_base_url)
    api_key = config.resolve_api_key()
    if backend == "openai" and not api_key:
        raise ValueError("embedding_backend=openai 需要 LLM_API_KEY（或 OPENAI_API_KEY）")
    if backend == "auto" and config.resolve_dashscope_key():
        return DashScopeEmbedder(config.resolve_dashscope_key(),
                                 config.dashscope_embedding_model,
                                 config.dashscope_base_url)
    if not api_key:
        raise ValueError(
            "需要 embedding 端点：设置 LLM_API_KEY 或 DASHSCOPE_API_KEY"
            "（测试环境请使用 tests/fake_openai.py 提供协议夹具）")
    return OpenAIEmbedder(config.embedding_base_url or config.llm_base_url,
                          api_key, config.embedding_model)


def cosine(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def pack(vec: List[float]) -> bytes:
    import array

    return array.array("f", vec).tobytes()


def unpack(blob: Optional[bytes], dim: int) -> Optional[List[float]]:
    if not blob:
        return None
    import array

    a = array.array("f")
    a.frombytes(blob)
    return list(a)

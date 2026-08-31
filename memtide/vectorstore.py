"""Vector store abstraction: Qdrant is the only production backend.

PostgreSQL remains the source of truth; Qdrant provides ANN vector search
and is rebuilt from PostgreSQL embeddings when necessary. The REST client uses
urllib directly — no vector SDK dependency.
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


class VectorStoreBase:
    def upsert(self, mem_id: str, vector: List[float], payload: Dict[str, Any]) -> None: ...
    def upsert_many(self, points: List[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        """Persist a batch of points; implementations should wait until the
        vector database acknowledges the write."""
        for mem_id, vector, payload in points:
            self.upsert(mem_id, vector, payload)
    def delete(self, mem_id: str) -> None: ...
    def clear(self) -> None: ...
    def search(self, query_vec: List[float], user_id: Optional[str] = None,
               agent_id: Optional[str] = None, run_id: Optional[str] = None,
               topk: int = 40) -> List[Tuple[str, float]]:
        """Returns [(memory_id, similarity)] sorted desc."""
        raise NotImplementedError


def _point_id(mem_id: str) -> int:
    return int.from_bytes(hashlib.blake2b(mem_id.encode(), digest_size=8).digest(), "big")


class QdrantVectorStore(VectorStoreBase):
    def __init__(self, url: str, collection: str, dim: int,
                 api_key: Optional[str] = None, timeout: float = 15.0):
        self.url = url.rstrip("/")
        self.collection = collection
        self.dim = dim
        self.api_key = api_key
        self.timeout = timeout
        self._ensure_collection()

    def _req(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{self.url}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        if self.api_key:
            req.add_header("api-key", self.api_key)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode() or "{}")

    def _ensure_collection(self) -> None:
        try:
            info = self._req("GET", f"/collections/{self.collection}")
            size = info.get("result", {}).get("config", {}).get("params", {}) \
                          .get("vectors", {}).get("size")
            if size and size != self.dim:
                print(f"[memtide] qdrant collection '{self.collection}' dim {size} "
                      f"!= embedder dim {self.dim}; recreating (run rebuild_index())")
                self._req("DELETE", f"/collections/{self.collection}")
                self._req("PUT", f"/collections/{self.collection}",
                          {"vectors": {"size": self.dim, "distance": "Cosine"}})
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
            self._req("PUT", f"/collections/{self.collection}",
                      {"vectors": {"size": self.dim, "distance": "Cosine"}})

    def upsert(self, mem_id, vector, payload):
        self._req("PUT", f"/collections/{self.collection}/points?wait=true",
                  {"points": [{"id": _point_id(mem_id), "vector": vector, "payload": payload}]})

    def upsert_many(self, points):
        """Batch upload with Qdrant's wait=true acknowledgement. Qdrant's
        async update queue was the source of false-successful rebuilds."""
        if not points:
            return
        for start in range(0, len(points), 256):
            batch = points[start:start + 256]
            self._req("PUT", f"/collections/{self.collection}/points?wait=true",
                      {"points": [{"id": _point_id(mid), "vector": vec, "payload": payload}
                                  for mid, vec, payload in batch]})

    def delete(self, mem_id):
        try:
            self._req("POST", f"/collections/{self.collection}/points/delete",
                      {"points": [_point_id(mem_id)]})
        except urllib.error.HTTPError:
            pass

    def count(self) -> int:
        """Exact point count used to validate index rebuild coverage."""
        res = self._req("POST", f"/collections/{self.collection}/points/count",
                        {"exact": True})
        return int(res.get("result", {}).get("count", 0))

    def clear(self):
        try:
            self._req("DELETE", f"/collections/{self.collection}")
        except urllib.error.HTTPError:
            pass
        self._ensure_collection()

    def search(self, query_vec, user_id=None, agent_id=None, run_id=None, topk=40):
        must: List[Dict[str, Any]] = []
        if user_id:
            must.append({"key": "user_id", "match": {"value": user_id}})
        if agent_id:
            must.append({"key": "agent_id", "match": {"value": agent_id}})
        if run_id:
            must.append({"key": "run_id", "match": {"value": run_id}})
        body: Dict[str, Any] = {"vector": query_vec, "limit": topk, "with_payload": True}
        if must:
            body["filter"] = {"must": must}
        res = self._req("POST", f"/collections/{self.collection}/points/search", body)
        return [(mid, float(p["score"]))
                for p in res.get("result", [])
                if (mid := (p.get("payload") or {}).get("memory_id"))]


def make_vector_store(config, storage, embedder) -> VectorStoreBase:
    if config.vector_backend != "qdrant":
        raise ValueError(
            f'vector_backend={config.vector_backend!r} 不支持：本项目只保留 Qdrant 生产向量库')
    dim = getattr(embedder, "dim", None)
    if dim is None:
        dim = len(embedder.embed("dimension probe"))
    return QdrantVectorStore(config.qdrant_url, config.qdrant_collection, dim,
                             api_key=getattr(config, "qdrant_api_key", None))

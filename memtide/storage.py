"""Storage contract: PostgreSQL is the only backend (production mode).

``StorageBase`` defines the interface the engine, retriever, consolidation
and REST layers talk to; ``PostgresStorage`` (pgstore.py) implements it
against PostgreSQL with pg_search (BM25). Embeddings are kept as a float32
bytea copy in relational storage for vector-index rebuilds.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .types import Event, Memory, utcnow


class StorageBase:
    """Contract implemented by the storage backend.

    Rows handed to ``Memory.from_row`` are dict-like with the memories-table
    column names; timestamps are ISO strings; ``get_embedding`` returns raw
    float32 bytes (or None).
    """

    def insert(self, mem: Memory, embedding: bytes) -> None: ...
    def replace_text(self, mem_id: str, new_text: str, entities: List[str],
                     embedding: bytes, metadata: Optional[Dict[str, Any]] = None) -> None: ...
    def soft_delete(self, mem_id: str) -> None: ...
    def hard_delete(self, mem_id: str) -> None: ...
    def supersede(self, mem_id: str, by_id: str) -> None: ...
    def mark_accessed(self, mem_ids: Iterable[str]) -> None: ...
    def log_event(self, memory_id: str, event: str, prev: Optional[str], new: Optional[str],
                  at: Optional[str] = None) -> None: ...
    def get_raw(self, mem_id: str) -> Optional[Any]: ...
    def get(self, mem_id: str) -> Optional[Memory]: ...
    def get_many(self, mem_ids: Iterable[str]) -> Dict[str, Memory]:
        """Batch get: one roundtrip for many ids (default loops get)."""
        out: Dict[str, Memory] = {}
        for mid in mem_ids:
            mem = self.get(mid)
            if mem is not None:
                out[mid] = mem
        return out
    def get_embedding(self, mem_id: str) -> Optional[bytes]: ...
    def get_embeddings(self, mem_ids: Iterable[str]) -> Dict[str, bytes]:
        """Batch embeddings: one roundtrip for many ids (default loops)."""
        out: Dict[str, bytes] = {}
        for mid in mem_ids:
            blob = self.get_embedding(mid)
            if blob is not None:
                out[mid] = blob
        return out
    def all_valid(self, user_id: Optional[str] = None, agent_id: Optional[str] = None,
                  run_id: Optional[str] = None) -> List[Memory]: ...
    def all_embeddings(self, user_id: Optional[str] = None, agent_id: Optional[str] = None,
                       run_id: Optional[str] = None) -> List[tuple]: ...
    def distinct_users(self) -> List[str]: ...
    def fts_search(self, query: str, limit: int = 40, user_id: Optional[str] = None) -> List[str]: ...
    def entity_lookup(self, entities: List[str], limit: int = 40) -> List[str]: ...
    def history(self, memory_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]: ...
    def stats(self) -> Dict[str, Any]: ...
    def all_rows(self, user_id: Optional[str] = None, agent_id: Optional[str] = None,
                 run_id: Optional[str] = None, include_invalid: bool = False) -> List[Memory]: ...
    def all_rows_with_embeddings(self) -> List[Dict[str, Any]]:  # for index rebuilds
        raise NotImplementedError
    def reset(self) -> None: ...
    def close(self) -> None: ...

    @staticmethod
    def _to_memory(row: Any) -> Memory:
        return Memory.from_row(dict(row))


# backwards-compatible alias (the contract used to be named Storage)
Storage = StorageBase


def make_storage(config) -> StorageBase:
    """PostgreSQL only. Any other storage_backend value is a config error."""
    if config.storage_backend != "postgres":
        raise ValueError(
            f'storage_backend={config.storage_backend!r} 不支持：只支持 PostgreSQL；'
            '请配置 MEMTIDE_PG_DSN 并设置 MEMTIDE_STORAGE=postgres')
    from .pgstore import PostgresStorage

    return PostgresStorage(config.pg_dsn)

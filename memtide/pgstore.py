"""PostgreSQL storage backend (psycopg 3) — implements the StorageBase contract.

- memories / entities / memory_history tables, auto-created on startup
- pg_search extension + BM25 index (ngram tokenizer) for CJK-capable full-text channel
- embeddings kept as float32 bytea (a copy for Qdrant index rebuilds;
  the live search path goes through the vector store, not this column)
- timestamps stored as ISO strings to keep Memory dataclass types unchanged

Install: pip install .  (psycopg is the only runtime dependency)
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from .storage import StorageBase
from .types import Event, Memory, utcnow

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    text          TEXT NOT NULL,
    memory_type   TEXT NOT NULL DEFAULT 'fact',
    user_id       TEXT NOT NULL DEFAULT 'default',
    agent_id      TEXT,
    run_id        TEXT,
    entities      TEXT NOT NULL DEFAULT '[]',
    metadata      TEXT NOT NULL DEFAULT '{}',
    importance    REAL NOT NULL DEFAULT 0.5,
    access_count  INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    valid_at      TEXT NOT NULL,
    invalid_at    TEXT,
    superseded_by TEXT,
    source        TEXT NOT NULL DEFAULT 'conversation',
    attachments   TEXT NOT NULL DEFAULT '[]',
    embedding     BYTEA
);
CREATE INDEX IF NOT EXISTS idx_mem_user  ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_mem_valid ON memories(invalid_at);

CREATE TABLE IF NOT EXISTS memory_history (
    seq        BIGSERIAL PRIMARY KEY,
    memory_id  TEXT NOT NULL,
    event      TEXT NOT NULL,
    prev_value TEXT,
    new_value  TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_hist_mem ON memory_history(memory_id);

CREATE TABLE IF NOT EXISTS entities (
    name      TEXT NOT NULL,
    memory_id TEXT NOT NULL,
    PRIMARY KEY (name, memory_id)
);
CREATE INDEX IF NOT EXISTS idx_ent_name ON entities(name);
"""


class PostgresStorage(StorageBase):
    def __init__(self, dsn: str):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                'PostgreSQL backend needs psycopg: pip install .'
            ) from e
        self.conn = psycopg.connect(dsn, row_factory=dict_row, autocommit=True)
        # SCHEMA public anchors extensions there: test runs point search_path
        # at a temp schema and must not relocate (or CASCADE-drop) shared
        # extensions when the temp schema is dropped
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
        self.conn.execute("CREATE EXTENSION IF NOT EXISTS pg_search SCHEMA public")
        self.conn.execute(_SCHEMA)
        self.conn.execute("DROP INDEX IF EXISTS idx_mem_trgm")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mem_bm25 ON memories USING bm25 (id, text) "
            "WITH (key_field='id', text_fields='{"
            '"text": {"tokenizer": {"type": "ngram", "min_gram": 2, "max_gram": 3, '
            '"prefix_only": false}, "filters": ["lowercase"]}}\')'
        )
        # migration for databases created before multimodal attachments existed
        self.conn.execute(
            "ALTER TABLE memories ADD COLUMN IF NOT EXISTS attachments TEXT NOT NULL DEFAULT '[]'"
        )

    # ---- writes --------------------------------------------------------------
    def insert(self, mem: Memory, embedding: bytes) -> None:
        import psycopg.types.json  # noqa: F401  (jsonb not used; TEXT columns)

        with self.conn.transaction():
            self.conn.execute(
                """INSERT INTO memories (id, text, memory_type, user_id, agent_id, run_id,
                                         entities, metadata, importance, access_count,
                                         last_accessed, created_at, updated_at, valid_at,
                                         invalid_at, superseded_by, source, attachments, embedding)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (mem.id, mem.text, mem.memory_type, mem.user_id, mem.agent_id, mem.run_id,
                 json.dumps(mem.entities, ensure_ascii=False),
                 json.dumps(mem.metadata, ensure_ascii=False),
                 mem.importance, mem.access_count, mem.last_accessed,
                 mem.created_at, mem.updated_at, mem.valid_at, mem.invalid_at,
                 mem.superseded_by, mem.source,
                 json.dumps(mem.attachments, ensure_ascii=False),
                 psycopg.Binary(embedding)),
            )
            for ent in mem.entities:
                self.conn.execute(
                    "INSERT INTO entities (name, memory_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (ent, mem.id),
                )
            self.log_event(mem.id, Event.ADD, None, mem.text, at=mem.created_at)

    def replace_text(self, mem_id: str, new_text: str, entities: List[str],
                     embedding: bytes, metadata: Optional[Dict[str, Any]] = None) -> None:
        row = self.get_raw(mem_id)
        if row is None:
            return
        if metadata is not None:
            merged = json.loads(row["metadata"] or "{}")
            merged.update(metadata)
            metadata_json = json.dumps(merged, ensure_ascii=False)
        else:
            metadata_json = row["metadata"]
        import psycopg

        with self.conn.transaction():
            self.conn.execute(
                """UPDATE memories SET text=%s, entities=%s, metadata=%s, embedding=%s,
                       updated_at=%s, valid_at=%s, invalid_at=NULL WHERE id=%s""",
                (new_text, json.dumps(entities, ensure_ascii=False), metadata_json,
                 psycopg.Binary(embedding), utcnow(), utcnow(), mem_id),
            )
            self.conn.execute("DELETE FROM entities WHERE memory_id = %s", (mem_id,))
            for ent in entities:
                self.conn.execute(
                    "INSERT INTO entities (name, memory_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (ent, mem_id),
                )
            self.log_event(mem_id, Event.UPDATE, row["text"], new_text)

    def soft_delete(self, mem_id: str) -> None:
        row = self.get_raw(mem_id)
        if row is None:
            return
        now = utcnow()
        with self.conn.transaction():
            self.conn.execute(
                "UPDATE memories SET invalid_at=%s, updated_at=%s WHERE id=%s", (now, now, mem_id)
            )
            self.log_event(mem_id, Event.DELETE, row["text"], None)

    def hard_delete(self, mem_id: str) -> None:
        row = self.get_raw(mem_id)
        if row is None:
            return
        with self.conn.transaction():
            self.conn.execute("DELETE FROM memories WHERE id = %s", (mem_id,))
            self.conn.execute("DELETE FROM entities WHERE memory_id = %s", (mem_id,))
            self.log_event(mem_id, Event.DELETE, row["text"], None)

    def supersede(self, mem_id: str, by_id: str) -> None:
        row = self.get_raw(mem_id)
        if row is None:
            return
        now = utcnow()
        with self.conn.transaction():
            self.conn.execute(
                "UPDATE memories SET invalid_at=%s, updated_at=%s, superseded_by=%s WHERE id=%s",
                (now, now, by_id, mem_id),
            )
            self.log_event(mem_id, Event.CONSOLIDATE, row["text"], by_id)

    def mark_accessed(self, mem_ids: Iterable[str]) -> None:
        now = utcnow()
        with self.conn.transaction():
            for mid in mem_ids:
                self.conn.execute(
                    """UPDATE memories SET access_count = access_count + 1,
                           last_accessed = %s WHERE id = %s""",
                    (now, mid),
                )
                self.log_event(mid, Event.ACCESS, None, None)

    def log_event(self, memory_id: str, event: str, prev: Optional[str],
                  new: Optional[str], at: Optional[str] = None) -> None:
        self.conn.execute(
            "INSERT INTO memory_history (memory_id, event, prev_value, new_value, created_at)"
            " VALUES (%s, %s, %s, %s, %s)",
            (memory_id, event, prev, new, at or utcnow()),
        )

    # ---- reads ------------------------------------------------------------
    def get_raw(self, mem_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM memories WHERE id = %s", (mem_id,)).fetchone()
        return dict(row) if row else None

    def get(self, mem_id: str) -> Optional[Memory]:
        row = self.get_raw(mem_id)
        return self._to_memory(row) if row else None

    def get_embedding(self, mem_id: str) -> Optional[bytes]:
        row = self.conn.execute(
            "SELECT embedding FROM memories WHERE id = %s", (mem_id,)
        ).fetchone()
        if row is None or row["embedding"] is None:
            return None
        return bytes(row["embedding"])

    def all_valid(self, user_id: Optional[str] = None, agent_id: Optional[str] = None,
                  run_id: Optional[str] = None) -> List[Memory]:
        clauses = ["invalid_at IS NULL"]
        params: List[Any] = []
        if user_id:
            clauses.append("user_id = %s")
            params.append(user_id)
        if agent_id:
            clauses.append("agent_id = %s")
            params.append(agent_id)
        if run_id:
            clauses.append("run_id = %s")
            params.append(run_id)
        rows = self.conn.execute(
            f"SELECT * FROM memories WHERE {' AND '.join(clauses)} ORDER BY created_at DESC",
            params,
        )
        return [self._to_memory(r) for r in rows]

    def all_embeddings(self, user_id: Optional[str] = None, agent_id: Optional[str] = None,
                       run_id: Optional[str] = None) -> List[tuple]:
        """(id, embedding_bytes) for all valid memories in scope — ONE query."""
        clauses = ["invalid_at IS NULL"]
        params: List[Any] = []
        for col, val in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)):
            if val:
                clauses.append(f"{col} = %s")
                params.append(val)
        rows = self.conn.execute(
            f"SELECT id, embedding FROM memories WHERE {' AND '.join(clauses)}", params)
        return [(r["id"], bytes(r["embedding"]) if r["embedding"] is not None else None)
                for r in rows]

    def distinct_users(self) -> List[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT user_id FROM memories WHERE invalid_at IS NULL")
        return [r["user_id"] for r in rows]

    def all_rows(self, user_id: Optional[str] = None, agent_id: Optional[str] = None,
                 run_id: Optional[str] = None, include_invalid: bool = False) -> List[Memory]:
        clauses, params = [], []
        if not include_invalid:
            clauses.append("invalid_at IS NULL")
        for col, val in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)):
            if val:
                clauses.append(f"{col} = %s")
                params.append(val)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.conn.execute(
            f"SELECT * FROM memories{where} ORDER BY created_at DESC", params)
        return [self._to_memory(r) for r in rows]

    def all_rows_with_embeddings(self) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE invalid_at IS NULL AND embedding IS NOT NULL")
        out = []
        for r in rows:
            d = dict(r)
            if d.get("embedding") is not None:
                d["embedding"] = bytes(d["embedding"])
            out.append(d)
        return out

    def fts_search(self, query: str, limit: int = 40, user_id: Optional[str] = None,
                   agent_id: Optional[str] = None, run_id: Optional[str] = None) -> List[str]:
        """BM25 full-text ranking (ParadeDB pg_search). ``paradedb.match``
        treats the query as plain text (no query syntax, OR over ngram terms),
        so apostrophes/punctuation in user input are safe and 2-char CJK
        queries work natively — no LIKE fallback needed."""
        q = query.strip()
        if not q:
            return []
        clauses = ["invalid_at IS NULL", "text @@@ paradedb.match('text', %s)"]
        params: List[Any] = [q]
        if user_id:
            clauses.append("user_id = %s")
            params.append(user_id)
        if agent_id:
            clauses.append("agent_id = %s")
            params.append(agent_id)
        if run_id:
            clauses.append("run_id = %s")
            params.append(run_id)
        params.append(limit)
        rows = self.conn.execute(
            f"""SELECT id FROM memories WHERE {' AND '.join(clauses)}
                ORDER BY paradedb.score(id) DESC LIMIT %s""",
            params,
        )
        ids = [r["id"] for r in rows]
        if not ids and len(q) < 2:
            # ngram min_gram=2: a 1-char query tokenizes to nothing and BM25
            # matches zero rows — fall back to substring ILIKE
            like_clauses = ["invalid_at IS NULL", "text ILIKE %s"]
            like_params: List[Any] = [f"%{q}%"]
            if user_id:
                like_clauses.append("user_id = %s")
                like_params.append(user_id)
            if agent_id:
                like_clauses.append("agent_id = %s")
                like_params.append(agent_id)
            if run_id:
                like_clauses.append("run_id = %s")
                like_params.append(run_id)
            like_params.append(limit)
            rows = self.conn.execute(
                f"SELECT id FROM memories WHERE {' AND '.join(like_clauses)} LIMIT %s",
                like_params,
            )
            ids = [r["id"] for r in rows]
        return ids

    def entity_lookup(self, entities: List[str], limit: int = 40,
                      user_id: Optional[str] = None, agent_id: Optional[str] = None,
                      run_id: Optional[str] = None) -> List[str]:
        def like_escape(s: str) -> str:
            return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

        out: List[str] = []
        for ent in entities:
            if len(ent) < 2:
                continue
            clauses = ["m.invalid_at IS NULL"]
            params: List[Any] = [f"%{like_escape(ent)}%"]
            if user_id:
                clauses.append("m.user_id = %s")
                params.append(user_id)
            if agent_id:
                clauses.append("m.agent_id = %s")
                params.append(agent_id)
            if run_id:
                clauses.append("m.run_id = %s")
                params.append(run_id)
            params.append(limit)
            rows = self.conn.execute(
                f"""SELECT DISTINCT e.memory_id FROM entities e
                   JOIN memories m ON m.id = e.memory_id
                   WHERE e.name ILIKE %s AND {' AND '.join(clauses)} LIMIT %s""",
                params,
            )
            out.extend(r["memory_id"] for r in rows)
        return list(dict.fromkeys(out))

    def history(self, memory_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        if memory_id:
            rows = self.conn.execute(
                "SELECT * FROM memory_history WHERE memory_id = %s ORDER BY seq DESC LIMIT %s",
                (memory_id, limit),
            )
        else:
            rows = self.conn.execute(
                "SELECT * FROM memory_history ORDER BY seq DESC LIMIT %s", (limit,)
            )
        return [dict(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        total = self.conn.execute(
            "SELECT COUNT(*) c FROM memories WHERE invalid_at IS NULL"
        ).fetchone()["c"]
        deleted = self.conn.execute(
            "SELECT COUNT(*) c FROM memories WHERE invalid_at IS NOT NULL"
        ).fetchone()["c"]
        events = {r["event"]: r["c"] for r in self.conn.execute(
            "SELECT event, COUNT(*) c FROM memory_history GROUP BY event"
        )}
        return {"active_memories": total, "invalidated_memories": deleted, "events": events}

    def reset(self) -> None:
        self.conn.execute("TRUNCATE memories, memory_history, entities")

    def close(self) -> None:
        self.conn.close()

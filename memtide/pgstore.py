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
import queue
import threading
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


class ConnectionLimitError(RuntimeError):
    """All pooled PG connections are busy (or PG refused more)."""


class PostgresStorage(StorageBase):
    def __init__(self, dsn: str, max_conns: int = 20):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                'PostgreSQL backend needs psycopg: pip install .'
            ) from e
        self._dsn = dsn
        self._row_factory = dict_row
        # psycopg connections are NOT thread-safe -> a bounded pool of
        # independent connections. The engine lock already serialises writes,
        # so peak concurrent use is low; 20 covers read paths (search/history/
        # stats) plus background adds without ever exhausting PG's cap.
        self._pool = queue.SimpleQueue()
        self._max_conns = max_conns
        self._guard = threading.Lock()
        self._opened = 0  # connections ever created; the pool's real bound
        # SCHEMA public anchors extensions there: test runs point search_path
        # at a temp schema and must not relocate (or CASCADE-drop) shared
        # extensions when the temp schema is dropped
        with self._acquire() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector SCHEMA public")
            conn.execute("CREATE EXTENSION IF NOT EXISTS pg_search SCHEMA public")
            conn.execute(_SCHEMA)
            conn.execute("DROP INDEX IF EXISTS idx_mem_trgm")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_bm25 ON memories USING bm25 (id, text) "
                "WITH (key_field='id', text_fields='{"
                '"text": {"tokenizer": {"type": "ngram", "min_gram": 2, "max_gram": 3, '
                '"prefix_only": false}, "filters": ["lowercase"]}}\')'
            )
            # migration for databases created before multimodal attachments existed
            conn.execute(
                "ALTER TABLE memories ADD COLUMN IF NOT EXISTS attachments TEXT NOT NULL DEFAULT '[]'"
            )

    def _acquire(self):
        """Context manager handing out a pooled connection (used via ``with``).

        All internal call sites use ``with self._acquire() as conn:`` — never
        leak a connection, so PG never sees "too many clients" again.
        """
        try:
            c = self._pool.get_nowait()
        except queue.Empty:
            # Pool empty: create a new connection only while under the cap.
            # The cap counts _opened (connections ever created), NOT the idle
            # queue — the queue is empty here by definition, so gating on
            # qsize() would never block anyone.
            under_cap = False
            with self._guard:
                under_cap = self._opened < self._max_conns
                if under_cap:
                    import psycopg

                    c = psycopg.connect(self._dsn, row_factory=self._row_factory,
                                        autocommit=True)
                    self._opened += 1
            if not under_cap:
                # wait up to 5s for a slot, then fail loudly with a clean error
                try:
                    c = self._pool.get(timeout=5)
                except queue.Empty:
                    raise ConnectionLimitError(
                        "all {} PG connections are busy; retry later".format(self._max_conns))

        class _Ctx:
            def __enter__(_self):
                return c

            def __exit__(_self, *exc):
                try:
                    if c.closed:
                        # dead connection: drop it and hand its slot back to
                        # the cap so a healthy replacement can be created
                        with self._guard:
                            self._opened -= 1
                        return False
                    self._pool.put(c)
                except Exception:
                    pass
                return False

        return _Ctx()

    # ---- writes --------------------------------------------------------------
    def insert(self, mem: Memory, embedding: bytes) -> None:
        import psycopg.types.json  # noqa: F401  (jsonb not used; TEXT columns)

        with self._acquire() as conn, conn.transaction():
            conn.execute(
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
                conn.execute(
                    "INSERT INTO entities (name, memory_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (ent, mem.id),
                )
            # same connection: the audit write must commit/roll back with the row
            self.log_event_conn(conn, mem.id, Event.ADD, None, mem.text, at=mem.created_at)

    def replace_text(self, mem_id: str, new_text: str, entities: List[str],
                     embedding: bytes, metadata: Optional[Dict[str, Any]] = None) -> None:
        row = self.get_raw(mem_id)
        if row is None:
            return
        if row.get("invalid_at") is not None:
            # Refuse to resurrect soft-deleted/superseded rows: replace_text
            # clears invalid_at, so updating one would silently bring it back.
            return
        if metadata is not None:
            merged = json.loads(row["metadata"] or "{}")
            merged.update(metadata)
            metadata_json = json.dumps(merged, ensure_ascii=False)
        else:
            metadata_json = row["metadata"]
        import psycopg

        with self._acquire() as conn, conn.transaction():
            conn.execute(
                """UPDATE memories SET text=%s, entities=%s, metadata=%s, embedding=%s,
                       updated_at=%s, valid_at=%s, invalid_at=NULL WHERE id=%s""",
                (new_text, json.dumps(entities, ensure_ascii=False), metadata_json,
                 psycopg.Binary(embedding), utcnow(), utcnow(), mem_id),
            )
            conn.execute("DELETE FROM entities WHERE memory_id = %s", (mem_id,))
            for ent in entities:
                conn.execute(
                    "INSERT INTO entities (name, memory_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (ent, mem_id),
                )
            self.log_event_conn(conn, mem_id, Event.UPDATE, row["text"], new_text)

    def soft_delete(self, mem_id: str) -> None:
        row = self.get_raw(mem_id)
        if row is None:
            return
        now = utcnow()
        with self._acquire() as conn, conn.transaction():
            conn.execute(
                "UPDATE memories SET invalid_at=%s, updated_at=%s WHERE id=%s", (now, now, mem_id)
            )
            self.log_event_conn(conn, mem_id, Event.DELETE, row["text"], None)

    def hard_delete(self, mem_id: str) -> None:
        row = self.get_raw(mem_id)
        if row is None:
            return
        with self._acquire() as conn, conn.transaction():
            conn.execute("DELETE FROM memories WHERE id = %s", (mem_id,))
            conn.execute("DELETE FROM entities WHERE memory_id = %s", (mem_id,))
            self.log_event_conn(conn, mem_id, Event.DELETE, row["text"], None)

    def supersede(self, mem_id: str, by_id: str) -> None:
        row = self.get_raw(mem_id)
        if row is None:
            return
        if row.get("invalid_at") is not None:
            # Already invalidated: keep the original superseded_by chain intact.
            return
        now = utcnow()
        with self._acquire() as conn, conn.transaction():
            conn.execute(
                "UPDATE memories SET invalid_at=%s, updated_at=%s, superseded_by=%s WHERE id=%s",
                (now, now, by_id, mem_id),
            )
            self.log_event_conn(conn, mem_id, Event.CONSOLIDATE, row["text"], by_id)

    def mark_accessed(self, mem_ids: Iterable[str]) -> None:
        ids = list(dict.fromkeys(mem_ids))
        if not ids:
            return
        now = utcnow()
        with self._acquire() as conn, conn.transaction():
            # single UPDATE for all hits (was one UPDATE + one INSERT per hit)
            conn.execute(
                """UPDATE memories SET access_count = access_count + 1,
                       last_accessed = %s WHERE id = ANY(%s)""",
                (now, ids),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO memory_history (memory_id, event, prev_value, new_value, created_at)"
                    " VALUES (%s, 'ACCESS', NULL, NULL, %s)",
                    [(mid, now) for mid in ids],
                )

    def log_event_conn(self, conn, memory_id: str, event: str, prev: Optional[str],
                       new: Optional[str], at: Optional[str] = None) -> None:
        conn.execute(
            "INSERT INTO memory_history (memory_id, event, prev_value, new_value, created_at)"
            " VALUES (%s, %s, %s, %s, %s)",
            (memory_id, event, prev, new, at or utcnow()),
        )

    def log_event(self, memory_id: str, event: str, prev: Optional[str],
                  new: Optional[str], at: Optional[str] = None) -> None:
        with self._acquire() as conn:
            self.log_event_conn(conn, memory_id, event, prev, new, at)

    # ---- reads ------------------------------------------------------------
    def get_raw(self, mem_id: str) -> Optional[Dict[str, Any]]:
        with self._acquire() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = %s", (mem_id,)).fetchone()
        return dict(row) if row else None

    def get(self, mem_id: str) -> Optional[Memory]:
        row = self.get_raw(mem_id)
        return self._to_memory(row) if row else None

    def get_many(self, mem_ids: Iterable[str]) -> Dict[str, Memory]:
        ids = list(dict.fromkeys(mem_ids))
        if not ids:
            return {}
        with self._acquire() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE id = ANY(%s)", (ids,))
        return {r["id"]: self._to_memory(r) for r in rows}

    def get_embedding(self, mem_id: str) -> Optional[bytes]:
        with self._acquire() as conn:
            row = conn.execute(
                "SELECT embedding FROM memories WHERE id = %s", (mem_id,)
            ).fetchone()
        if row is None or row["embedding"] is None:
            return None
        return bytes(row["embedding"])

    def get_embeddings(self, mem_ids: Iterable[str]) -> Dict[str, bytes]:
        ids = list(dict.fromkeys(mem_ids))
        if not ids:
            return {}
        out: Dict[str, bytes] = {}
        with self._acquire() as conn:
            for r in conn.execute(
                    "SELECT id, embedding FROM memories WHERE id = ANY(%s)", (ids,)):
                if r["embedding"] is not None:
                    out[r["id"]] = bytes(r["embedding"])
        return out

    def all_valid(self, user_id: Optional[str] = None, agent_id: Optional[str] = None,
                  run_id: Optional[str] = None,
                  limit: Optional[int] = None) -> List[Memory]:
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
        sql = f"SELECT * FROM memories WHERE {' AND '.join(clauses)} ORDER BY created_at DESC"
        if limit is not None:
            # push the cap into PG: slicing in Python would first load every
            # row in scope (attachments JSON included)
            sql += " LIMIT %s"
            params.append(limit)
        with self._acquire() as conn:
            rows = conn.execute(sql, params)
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
        with self._acquire() as conn:
            rows = conn.execute(
                f"SELECT id, embedding FROM memories WHERE {' AND '.join(clauses)}", params)
        return [(r["id"], bytes(r["embedding"]) if r["embedding"] is not None else None)
                for r in rows]

    def distinct_users(self) -> List[str]:
        with self._acquire() as conn:
            rows = conn.execute(
                "SELECT DISTINCT user_id FROM memories WHERE invalid_at IS NULL")
        return [r["user_id"] for r in rows]

    def all_rows(self, user_id: Optional[str] = None, agent_id: Optional[str] = None,
                 run_id: Optional[str] = None, include_invalid: bool = False,
                 limit: Optional[int] = None) -> List[Memory]:
        clauses, params = [], []
        if not include_invalid:
            clauses.append("invalid_at IS NULL")
        for col, val in (("user_id", user_id), ("agent_id", agent_id), ("run_id", run_id)):
            if val:
                clauses.append(f"{col} = %s")
                params.append(val)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM memories{where} ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT %s"
            params.append(limit)
        with self._acquire() as conn:
            rows = conn.execute(sql, params)
        return [self._to_memory(r) for r in rows]

    def all_rows_with_embeddings(self) -> List[Dict[str, Any]]:
        with self._acquire() as conn:
            rows = conn.execute(
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
        with self._acquire() as conn:
            rows = conn.execute(
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
                rows = conn.execute(
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
        with self._acquire() as conn:
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
                rows = conn.execute(
                    f"""SELECT DISTINCT e.memory_id FROM entities e
                       JOIN memories m ON m.id = e.memory_id
                       WHERE e.name ILIKE %s AND {' AND '.join(clauses)} LIMIT %s""",
                    params,
                )
                out.extend(r["memory_id"] for r in rows)
        return list(dict.fromkeys(out))

    def history(self, memory_id: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        with self._acquire() as conn:
            if memory_id:
                rows = conn.execute(
                    "SELECT * FROM memory_history WHERE memory_id = %s ORDER BY seq DESC LIMIT %s",
                    (memory_id, limit),
                )
            else:
                rows = conn.execute(
                    "SELECT * FROM memory_history ORDER BY seq DESC LIMIT %s", (limit,)
                )
        return [dict(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        with self._acquire() as conn:
            total = conn.execute(
                "SELECT COUNT(*) c FROM memories WHERE invalid_at IS NULL"
            ).fetchone()["c"]
            deleted = conn.execute(
                "SELECT COUNT(*) c FROM memories WHERE invalid_at IS NOT NULL"
            ).fetchone()["c"]
            events = {r["event"]: r["c"] for r in conn.execute(
                "SELECT event, COUNT(*) c FROM memory_history GROUP BY event"
            )}
            by_type = {r["memory_type"]: r["c"] for r in conn.execute(
                "SELECT memory_type, COUNT(*) c FROM memories WHERE invalid_at IS NULL "
                "GROUP BY memory_type")}
            gate = {r["g"]: r["c"] for r in conn.execute(
                "SELECT metadata::jsonb->>'gate' AS g, COUNT(*) c FROM memories "
                "WHERE invalid_at IS NULL GROUP BY 1")}
        return {"active_memories": total, "invalidated_memories": deleted, "events": events,
                "by_type": by_type, "by_gate": gate}

    def reset(self) -> None:
        with self._acquire() as conn:
            conn.execute("TRUNCATE memories, memory_history, entities")

    def close(self) -> None:
        # close every pooled connection that is currently idle
        while True:
            try:
                c = self._pool.get_nowait()
            except Exception:
                break
            try:
                c.close()
            except Exception:
                pass

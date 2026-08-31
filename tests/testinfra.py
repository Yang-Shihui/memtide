"""Test infrastructure: fake OpenAI server + isolated PostgreSQL schemas.

Every test gets its own engine bound to a fresh PG schema (tables created
inside it, dropped on process exit) against the real test database, plus a
local fake OpenAI-compatible server for LLM/embedding calls. Nothing
touches the production schema and no network leaves the machine.

Required: a reachable PostgreSQL (MEMTIDE_TEST_PG_DSN or the default
postgresql://mnemos:mnemos-local-dev@localhost:5432/mnemos — the compose
stack publishes 5432).
"""

from __future__ import annotations

import atexit
import os
import uuid
import urllib.error
import urllib.request

import psycopg

TEST_DSN = (os.environ.get("MEMTIDE_TEST_PG_DSN")
            or "postgresql://mnemos:mnemos-local-dev@localhost:5432/mnemos")

_pending_drops: list = []
_pending_collections: list[str] = []
QDRANT_URL = os.environ.get("MEMTIDE_TEST_QDRANT_URL", "http://localhost:6333")


def _drop_all() -> None:
    if not _pending_drops:
        return
    try:
        admin = psycopg.connect(TEST_DSN, autocommit=True)
    except Exception:
        return
    for schema in _pending_drops:
        try:
            admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        except Exception:
            pass
    admin.close()
    for collection in _pending_collections:
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"{QDRANT_URL}/collections/{collection}", method="DELETE"), timeout=10)
        except Exception:
            pass


def new_collection() -> str:
    collection = "test_" + uuid.uuid4().hex[:16]
    _pending_collections.append(collection)
    return collection


def new_schema_dsn() -> tuple[str, str]:
    """Create a unique schema; return (dsn-with-search_path, schema).

    All created schemas are dropped in one batch when the test process
    exits."""
    schema = "t" + uuid.uuid4().hex[:12]
    admin = psycopg.connect(TEST_DSN, autocommit=True)
    admin.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    admin.close()
    _pending_drops.append(schema)
    sep = "&" if "?" in TEST_DSN else "?"
    return (f"{TEST_DSN}{sep}options=-c%20search_path%3D{schema}%2Cpublic", schema)


atexit.register(_drop_all)

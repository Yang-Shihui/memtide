# Contributing to Memtide

Thanks for considering a contribution! Memtide is a young project — bug
reports and focused PRs are both very welcome.

## Development setup

```bash
git clone <your-fork> && cd memtide
pip install .                # psycopg is the only runtime dependency
cp .env.example .env         # fill in LLM / embedding keys for live scripts
docker compose up -d postgres qdrant   # tests need reachable PG + Qdrant
python -m unittest tests.test_memtide   # 73 hermetic tests, ~15s, no network
```

Live integration tests (`tests/test_live.py`, 8 tests, real LLM/embedding
endpoints) are gated behind `MEMTIDE_LIVE=1` and skipped by default.
`scripts/live_check.py` exercises a real deployment — it always creates its
own PG schema and Qdrant collection, never touching production data.

## Ground rules

- **Production path only**: PostgreSQL + Qdrant + a real OpenAI-compatible
  LLM/embedding endpoint. PRs adding SQLite/mock/local-vector backends will
  be declined — hermetic tests use the local OpenAI-protocol server in
  `tests/fake_openai.py` instead.
- **PostgreSQL is the source of truth**, Qdrant is a rebuildable index.
  Every write goes through the audit log; deletes are soft by default.
- **No silent fallbacks**: if an infrastructure dependency is down, fail
  loudly.
- Keep the core stdlib-only (psycopg excepted); Qdrant access is plain
  urllib REST.
- Match the existing test style — every bug fix lands with a regression
  test in `tests/test_memtide.py`.

## Commit style

Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`).

## Reporting issues

Please include: Python version, how you deployed (compose / pip / source),
relevant `/stats` output, and — for search-quality reports — the full
`components` breakdown of a few hits. **Never paste API keys or `.env`
contents.**

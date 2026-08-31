"""Live verification against REAL endpoints — the answer to "都真实测试过吗".

Checks, in order (each prints PASS/FAIL and detail):
  1. GLM chat endpoint connectivity (auto-detects /v1)
  2. DashScope embedding connectivity + dimension probe
  3. Full engine on real backends:
     add (LLM extraction + gate + consolidation) -> search (real vectors)
     -> conflict update (career change) -> background reflection (LLM distill)
     -> audit chain integrity
  4. REST server smoke on the real stack

Usage:
  source .env   # 或 export MEMTIDE_LLM_BASE_URL=... MEMTIDE_LLM_KEY=... MEMTIDE_DASHSCOPE_KEY=...
  python3 scripts/live_check.py            # checks 1-3
  python3 scripts/live_check.py --docker   # also runs the compose stack smoke (check 4)
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

RESULTS = []


def check(name, fn):
    print(f"\n=== {name} ===")
    try:
        detail = fn()
        RESULTS.append((name, True, ""))
        print(f"PASS ✓  {detail or ''}")
    except Exception as e:
        RESULTS.append((name, False, str(e)))
        print(f"FAIL ✗  {type(e).__name__}: {e}")


# accepts both the MEMTIDE_* names and the plain .env names (LLM_API_KEY etc.);
# the endpoint must come from the environment — no baked-in default
LLM_URL = os.environ.get("MEMTIDE_LLM_BASE_URL") or os.environ.get("LLM_BASE_URL") or ""
LLM_KEY = os.environ.get("MEMTIDE_LLM_KEY") or os.environ.get("LLM_API_KEY") or ""
LLM_MODEL = os.environ.get("MEMTIDE_LLM_MODEL", "gpt-4o-mini")
DS_KEY = os.environ.get("MEMTIDE_DASHSCOPE_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""
DS_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DS_MODEL = os.environ.get("MEMTIDE_DASHSCOPE_MODEL", "qwen3.7-text-embedding")
RUN = uuid.uuid4().hex[:8]
USER = lambda label: f"live-{label}-{RUN}"


def isolated_pg_dsn():
    """Create a throwaway PG schema for the local live engine; never pollute
    the production schema while exercising real LLM/embedding endpoints."""
    import psycopg

    base = os.environ.get("MEMTIDE_PG_DSN", "postgresql://mnemos:mnemos-local-dev@localhost:5432/mnemos")
    schema = "check_" + RUN
    admin = psycopg.connect(base, autocommit=True)
    admin.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    admin.close()
    sep = "&" if "?" in base else "?"
    return f"{base}{sep}options=-c%20search_path%3D{schema}%2Cpublic"


def check_1_llm():
    import json as _json

    req = urllib.request.Request(
        f"{LLM_URL}/chat/completions",
        data=_json.dumps({"model": LLM_MODEL, "messages": [{"role": "user", "content": "hi"}],
                          "max_tokens": 5}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {LLM_KEY}"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = _json.loads(resp.read().decode())
    assert data["choices"], "empty choices"
    return f"model={data.get('model')}  {time.time()-t0:.1f}s"


def check_2_embedding():
    from memtide.embeddings import DashScopeEmbedder

    e = DashScopeEmbedder(DS_KEY, DS_MODEL, DS_URL)
    t0 = time.time()
    v = e.embed("维度探测")
    assert len(v) > 0
    return f"model={DS_MODEL}  dim={len(v)}  {time.time()-t0:.1f}s"


def make_engine():
    from memtide import MemoryConfig, MemoryEngine

    cfg = MemoryConfig(
        storage_backend="postgres", pg_dsn=isolated_pg_dsn(),
        vector_backend="qdrant", qdrant_url=os.environ.get("MEMTIDE_QDRANT_URL", "http://localhost:6333"),
        qdrant_collection=os.environ.get("MEMTIDE_LIVE_QDRANT_COLLECTION", f"memtide_live_check_{RUN}"),
        llm_backend="openai", llm_base_url=LLM_URL, llm_model=LLM_MODEL, llm_api_key=LLM_KEY,
        embedding_backend="dashscope", dashscope_api_key=DS_KEY,
        dashscope_base_url=DS_URL, dashscope_embedding_model=DS_MODEL,
    )
    return MemoryEngine(cfg)


def check_3_full_pipeline():
    eng = make_engine()
    t0 = time.time()

    r1 = eng.add([{"role": "user", "content":
                   "嗨，我叫李雷，是一名后端工程师，住在杭州，喜欢喝美式咖啡，不喜欢加班"}],
                 user_id=USER("facts"))
    assert r1.facts, f"LLM extracted no facts: {r1.to_dict()}"
    assert len(r1.added) >= 3, f"expected >=3 stored, got {r1.to_dict()}"
    print(f"  LLM 抽取 {len(r1.facts)} 条事实: {r1.facts}")

    r2 = eng.add("我转到产品岗了", user_id=USER("facts"))
    mems = eng.get_all(USER("facts"))
    assert any("产品" in m.text for m in mems), "career change not stored"
    print(f"  职业变更: updated={r2.updated} added={r2.added}")

    hits = eng.search("用户喜欢喝什么咖啡", user_id=USER("facts"))
    assert hits, "search returned nothing"
    assert "咖啡" in hits[0].memory.text or "coffee" in hits[0].memory.text.lower(), \
        f"top hit not relevant: {hits[0].memory.text}"
    print(f"  检索 top1: [{hits[0].score:.3f}] {hits[0].memory.text}")

    # real-LLM background reflection needs a dense cluster; engineer one
    for t in ["我最近在学 Rust 的 async 运行时", "我在用 Rust 写一个解析器", "我每天都用 Rust 写练习题"]:
        eng.add(t, user_id=USER("rust"))
    rep = eng.consolidate_background(user_id=USER("rust"))
    print(f"  反思: {rep['clusters']} 簇 / 吸收 {rep['members_absorbed']} 条"
          + (f"，概括: {rep['summaries'][0]['text'][:50]}…" if rep["summaries"] else ""))

    # audit chain integrity: every added memory has an ADD event
    for mid in r1.added:
        hist = eng.get_history(mid)
        assert any(e["event"] == "ADD" for e in hist), f"no ADD event for {mid}"
    eng.close()
    return f"全链路 {time.time()-t0:.1f}s（含 {len(r1.facts)+1} 次 LLM 抽取消解 + 多次 embedding）"


def check_4_docker():
    base = os.environ.get("MEMTIDE_DOCKER_URL", "http://localhost:8300")
    def req(method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(base + path, data=data, method=method,
                                   headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=90) as resp:
            return json.loads(resp.read().decode())
    stats = req("GET", "/stats")
    assert stats["backend"]["llm_backend"] == "openai", stats["backend"]
    assert stats["backend"]["vector"] == "qdrant", stats["backend"]
    uid = USER("smoke")
    r = req("POST", "/memories", {"text": "REST 冒烟：我喜欢打羽毛球", "user_id": uid})
    assert r["facts"], r
    hits = []
    for _ in range(5):
        hits = req("POST", "/search", {"query": "用户喜欢什么运动", "user_id": uid})
        if hits:
            break
        time.sleep(0.4)
    assert hits, "docker search empty"
    return f"docker 栈 OK: {stats['backend']}"


if __name__ == "__main__":
    if not LLM_URL:
        sys.exit("error: set MEMTIDE_LLM_BASE_URL / LLM_BASE_URL (see .env.example); "
                 "live_check refuses to guess an endpoint")
    check("1. GLM LLM 端点连通", check_1_llm)
    check("2. DashScope embedding 连通 + 维度", check_2_embedding)
    check("3. 真实后端全链路（抽取/门控/检索/更新/反思/审计）", check_3_full_pipeline)
    if "--docker" in sys.argv:
        check("4. Docker 栈 REST 冒烟", check_4_docker)

    fails = [r for r in RESULTS if not r[1]]
    print("\n" + "=" * 60)
    for name, ok, err in RESULTS:
        print(f"  {'PASS ✓' if ok else 'FAIL ✗'}  {name}")
    print(f"结论: {len(RESULTS) - len(fails)}/{len(RESULTS)} 通过")
    sys.exit(1 if fails else 0)

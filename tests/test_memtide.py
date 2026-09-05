import base64
import os
import sys
import tempfile
import time
import unittest
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memtide import MemoryConfig, MemoryEngine, MemoryType
from memtide.decay import retention
from memtide.pgstore import ConnectionLimitError
from tests.fake_openai import start_fake_server
from tests.testinfra import new_collection, new_schema_dsn

# one fake OpenAI-compatible server per test process: the engine speaks the
# real protocol, the fake reproduces the old deterministic rules at the
# HTTP boundary — hermetic (no network, no keys) and fast
FAKE_URL = start_fake_server()


def cfg(**kw):
    dsn, _ = new_schema_dsn()
    base = dict(storage_backend="postgres", pg_dsn=dsn,
                vector_backend="qdrant", qdrant_url="http://localhost:6333",
                qdrant_collection=new_collection(),
                llm_backend="openai", llm_base_url=FAKE_URL, llm_api_key="test",
                embedding_backend="openai", embedding_base_url=FAKE_URL,
                embedding_model="fake-embedder")
    base.update(kw)
    return MemoryConfig(**base)


def fresh_engine(**kw):
    """Engine bound to its own isolated PG schema (dropped at process exit)."""
    return MemoryEngine(cfg(**kw))


def make_test_png(w=64, h=64, rgb=(200, 60, 90)):
    """A real PNG (solid color) built with the stdlib — big enough that vision
    models accept it (they reject images below ~10px on a side)."""
    import struct
    import zlib

    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


class TestWritePipeline(unittest.TestCase):
    def setUp(self):
        self.eng = fresh_engine()

    def test_extracts_atomic_facts_zh(self):
        res = self.eng.add("用户：我叫李雷，住在杭州，喜欢喝美式咖啡，不喜欢加糖", user_id="u1")
        self.assertGreater(len(res.facts), 2, res.facts)
        texts = " | ".join(res.facts)
        self.assertIn("的名字是李雷", texts)
        self.assertIn("住在杭州", texts)
        self.assertIn("喜欢喝美式咖啡", texts)
        self.assertGreaterEqual(len(res.added), 3)

    def test_extracts_atomic_facts_en(self):
        res = self.eng.add("Hi! My name is Alice and I work at Acme Corp. I love hiking.")
        texts = " | ".join(res.facts)
        self.assertIn("name is Alice", texts)
        self.assertIn("works at Acme Corp", texts)
        self.assertIn("likes hiking", texts)

    def test_exact_duplicate_is_noop(self):
        self.eng.add("我叫李雷", user_id="u1")
        before = len(self.eng.get_all("u1"))
        res = self.eng.add("我叫李雷", user_id="u1")
        after = len(self.eng.get_all("u1"))
        self.assertEqual(after, before)
        self.assertGreaterEqual(res.noop, 1)

    def test_update_supersedes_near_duplicate(self):
        self.eng.add("我叫李雷", user_id="u1")
        res = self.eng.add("我叫李雷峰，大家可以叫我老李", user_id="u1")
        mems = self.eng.get_all("u1")
        # near-duplicate should be UPDATEd, not duplicated (tolerant assert)
        self.assertLessEqual(len(mems), 3)
        self.assertTrue(res.updated or res.noop or res.added)

    def test_polarity_conflict_updates(self):
        self.eng.add("我喜欢喝咖啡", user_id="u2")
        res = self.eng.add("我讨厌喝咖啡", user_id="u2")
        mems = self.eng.get_all("u2")
        likes = [m for m in mems if "喜欢" in m.text and "不喜欢" not in m.text]
        dislikes = [m for m in mems if "不喜欢" in m.text or "讨厌" in m.text]
        self.assertEqual(len(dislikes), 1)
        # conflict resolution should have updated/replaced the old polarity
        self.assertTrue(res.updated or len(likes) == 0,
                        "old 'likes coffee' memory should be superseded")

    def test_volatile_slot_update_tracks_history(self):
        self.eng.add("我叫李雷，住在杭州", user_id="u3")
        res = self.eng.add("我搬到上海了", user_id="u3")
        self.assertEqual(len(res.updated), 1)
        mems = self.eng.get_all("u3")
        self.assertTrue(any(m.text == "用户住在上海" for m in mems))
        self.assertFalse(any("杭州" in m.text for m in mems))
        # audit trail: ADD -> UPDATE chain on the surviving memory
        hist = self.eng.get_history([m for m in mems if m.text == "用户住在上海"][0].id)
        events = [e["event"] for e in hist]
        self.assertIn("UPDATE", events)
        self.assertIn("ADD", events)

    def test_slot_canonicalize_aliases(self):
        from memtide.slots import canonicalize_slot, same_slot

        self.assertEqual(canonicalize_slot("city"), "location")
        self.assertEqual(canonicalize_slot("住址"), "location")
        self.assertEqual(canonicalize_slot("  Location "), "location")
        self.assertEqual(canonicalize_slot("spouse"), "spouse")  # open slot passes through
        self.assertIsNone(canonicalize_slot("!!!"))
        self.assertIsNone(canonicalize_slot("like"))  # multi-value -> no hint
        self.assertTrue(same_slot("city", "location"))
        self.assertTrue(same_slot("住址", "location"))
        self.assertFalse(same_slot("location", "role"))

    def test_gate_alias_volatile_update(self):
        from memtide.gating import PredictiveGate
        from memtide.types import ExtractedFact, Memory

        mem = Memory(text="用户住在杭州", user_id="x", metadata={"slot": "location"})
        fact = ExtractedFact(text="用户住在上海", memory_type="fact", slot="city")
        d = PredictiveGate(cfg()).evaluate(fact, [(0.6, mem)])
        self.assertTrue(d.store)
        self.assertEqual(d.reason, "volatile-update")

    def test_pg_guards_no_resurrect_or_rechain(self):
        from memtide.types import Memory

        m = Memory(text="用户住在杭州", user_id="g", metadata={"slot": "location"})
        emb = self.eng.retriever.embed_for_storage(m.text)
        self.eng.store.insert(m, emb)
        self.eng.store.soft_delete(m.id)
        # replace_text must refuse to resurrect an invalidated row
        self.eng.store.replace_text(m.id, "用户住在上海", [], emb)
        self.assertIsNotNone(self.eng.store.get(m.id).invalid_at)
        # supersede twice must keep the original chain
        m2 = Memory(text="用户住在上海", user_id="g", metadata={"slot": "location"})
        emb2 = self.eng.retriever.embed_for_storage(m2.text)
        self.eng.store.insert(m2, emb2)
        self.eng.store.supersede(m.id, m2.id)
        first = self.eng.store.get(m.id).superseded_by
        m3 = Memory(text="用户住在苏州", user_id="g")
        emb3 = self.eng.retriever.embed_for_storage(m3.text)
        self.eng.store.insert(m3, emb3)
        self.eng.store.supersede(m.id, m3.id)
        self.assertEqual(self.eng.store.get(m.id).superseded_by, first)

    def test_ssrf_media_hosts_blocked(self):
        from memtide import multimodal

        for bad in ("http://localhost:6333/collections",
                    "http://127.0.0.1:5432/x",
                    "http://169.254.169.254/latest/meta-data/",
                    "http://10.0.0.5/secret",
                    # non-canonical IPv4 literals libc routes to loopback but
                    # ipaddress.ip_address cannot parse (regression: the old
                    # guard treated the parse failure as "DNS name, allowed")
                    "http://127.1:6333/collections",
                    "http://2130706433/",
                    "http://0x7f.0.0.1:6333/",
                    "http://127.0.1/x"):
            with self.assertRaises(ValueError, msg=bad):
                multimodal._guard_url(bad)
        # public hosts pass the guard (no fetch performed here; a literal
        # keeps this hermetic — no DNS)
        multimodal._guard_url("https://93.184.216.34/photo.jpg")

    def test_manual_add_no_infer(self):
        res = self.eng.add("记住：部署密钥放在 1password 的 infra 库里", infer=False)
        self.assertEqual(res.facts, ["记住：部署密钥放在 1password 的 infra 库里"])
        self.assertEqual(len(res.added), 1)

    def tearDown(self):
        self.eng.close()


class TestRetrieval(unittest.TestCase):
    def setUp(self):
        self.eng = fresh_engine()

    def _seed(self, user="u1"):
        self.eng.add("我叫李雷，住在杭州，喜欢喝美式咖啡", user_id=user)
        self.eng.add("我是后端工程师，在一家电商公司工作", user_id=user)
        self.eng.add("我正在学习 Rust 和 WASM", user_id=user)

    def test_semantic_hit(self):
        self._seed()
        hits = self.eng.search("用户住在哪个城市？", user_id="u1")
        self.assertTrue(hits)
        self.assertIn("杭州", hits[0].memory.text)
        self.assertGreater(hits[0].components["semantic"], 0.3)

    def test_user_scoping(self):
        self._seed(user="a")
        self.eng.add("我叫王五，住在上海", user_id="b")
        hits_a = self.eng.search("用户住在哪里？", user_id="a")
        self.assertIn("杭州", hits_a[0].memory.text)
        hits_b = self.eng.search("用户住在哪里？", user_id="b")
        self.assertIn("上海", hits_b[0].memory.text)

    def test_rrf_explains_scores(self):
        self._seed()
        hits = self.eng.search("rust", user_id="u1")
        self.assertTrue(all("components" in h.to_dict() for h in hits))
        self.assertIn("Rust", hits[0].memory.text)

    def test_deleted_memory_not_retrieved(self):
        self._seed()
        hits = self.eng.search("用户喜欢什么咖啡", user_id="u1")
        target = hits[0].memory.id
        self.eng.delete(target)  # soft delete
        hits2 = self.eng.search("用户喜欢什么咖啡", user_id="u1")
        self.assertTrue(all(h.memory.id != target for h in hits2))
        # but history keeps the audit trail (Zep-style)
        hist = self.eng.get_history(target)
        self.assertTrue(any(e["event"] == "DELETE" for e in hist))

    def test_reinforcement_on_retrieval(self):
        self._seed()
        q = "用户喜欢什么咖啡"
        h1 = self.eng.search(q, user_id="u1")[0]
        n1 = h1.memory.access_count
        h2 = self.eng.search(q, user_id="u1")[0]
        self.assertEqual(h2.memory.access_count, n1 + 1)

    def tearDown(self):
        self.eng.close()


class TestDecayAndContext(unittest.TestCase):
    def test_retention_math(self):
        eng = fresh_engine()
        eng.add("我叫李雷")
        mem = eng.get_all()[0]
        self.assertGreaterEqual(retention(mem, 45.0, 0.4), 0.999)
        eng.close()

    def test_forgotten_memory_hidden_but_recoverable(self):
        from datetime import datetime, timedelta, timezone

        eng = fresh_engine(half_life_days=5, retention_floor=0.05)
        eng.add("我叫李雷，住在杭州")
        mem = eng.get_all()[0]
        # age the memory to 100 days -> retention ~ 0.5^20 << floor
        old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat(timespec="seconds")
        with eng.store._acquire() as conn:
            conn.execute(
                "UPDATE memories SET created_at=%s, updated_at=%s, valid_at=%s WHERE id=%s",
                (old, old, old, mem.id),
            )
        self.assertLess(retention(eng.store.get(mem.id), 5.0, 0.4), 0.05)
        hits = eng.search("用户住哪", include_forgotten=False)
        self.assertTrue(all(h.memory.id != mem.id for h in hits))
        hits_f = eng.search("用户住哪", include_forgotten=True)
        self.assertTrue(any(h.memory.id == mem.id for h in hits_f))
        eng.close()

    def test_render_context(self):
        eng = fresh_engine()
        eng.add("我叫李雷，是一名后端工程师，住在杭州，喜欢喝美式咖啡，不喜欢加班")
        block = eng.render_context(user_id="default", query="用户喜欢什么？")
        self.assertIn("## Memory", block)
        self.assertIn("的名字是李雷", block)
        self.assertIn("## Relevant to current query", block)
        eng.close()


class TestPersistenceAndCli(unittest.TestCase):
    def test_persist_across_engines(self):
        cfgx = cfg()
        e1 = MemoryEngine(cfgx)
        e1.add("我叫李雷，住在杭州")
        e1.close()
        e2 = MemoryEngine(cfgx)  # same PG schema -> data persists
        mems = e2.get_all("default")
        self.assertTrue(any("李雷" in m.text for m in mems))
        self.assertTrue(any("住在杭州" in m.text for m in mems))
        e2.close()

    def test_cli_smoke(self):
        import os as _os

        from memtide.cli import main

        dsn, _ = new_schema_dsn()
        env = {**_os.environ, "MEMTIDE_PG_DSN": dsn,
               "LLM_BASE_URL": FAKE_URL, "LLM_API_KEY": "test",
               "LLM_BACKEND": "openai", "EMBEDDING_BACKEND": "openai",
               "EMBEDDING_BASE_URL": FAKE_URL}
        saved = {k: _os.environ.get(k) for k in env}
        _os.environ.update(env)
        try:
            rc = main(["add", "我叫李雷，喜欢 Rust"])
            self.assertEqual(rc, 0)
            rc = main(["search", "用户的名字"])
            self.assertEqual(rc, 0)
            rc = main(["context", "用户喜欢什么"])
            self.assertEqual(rc, 0)
            rc = main(["stats"])
            self.assertEqual(rc, 0)
        finally:
            for k, v in saved.items():
                if v is None:
                    _os.environ.pop(k, None)
                else:
                    _os.environ[k] = v


class TestPredictiveGate(unittest.TestCase):
    """Predictive-coding gating: encode only prediction error."""

    def test_gate_rejects_fully_predicted_facts(self):
        eng = fresh_engine()
        eng.add("我叫李雷，住在杭州", user_id="g1")
        n_before = len(eng.get_all("g1"))
        r2 = eng.add("我叫李雷，住在杭州", user_id="g1")
        self.assertEqual(len(r2.rejected), 2)
        self.assertGreaterEqual(r2.noop, 2)
        self.assertEqual(len(eng.get_all("g1")), n_before)
        # decisions are auditable
        self.assertIn("redundant", [d["reason"] for d in r2.gate.values()])
        eng.close()

    def test_gate_routes_novel_with_importance_boost(self):
        eng = fresh_engine()
        r = eng.add("我叫李雷，住在杭州，喜欢喝美式咖啡", user_id="g2")
        self.assertEqual(r.rejected, [])
        mem = eng.get(r.added[0])
        self.assertEqual(mem.metadata.get("gate"), "novel")
        self.assertGreater(mem.metadata.get("surprise_bits", 0), 2.5)
        self.assertGreater(mem.importance, 0.95)  # name base 0.95 + boost, capped
        eng.close()

    def test_gate_decision_boundaries(self):
        from memtide.gating import PredictiveGate
        from memtide.types import ExtractedFact, Memory

        gate = PredictiveGate(cfg())
        fact = ExtractedFact(text="用户喜欢爵士乐")

        def decide(sims):
            mems = [(s, Memory(text="x", metadata={"slot": None})) for s in sims]
            return gate.evaluate(fact, mems)

        # identical fact already known -> reject (fully predicted)
        self.assertFalse(decide([1.0]).store)
        # schema-congruent variant -> integrate, no boost
        d = decide([0.7, 0.6])
        self.assertTrue(d.store)
        self.assertEqual(d.reason, "integrate")
        self.assertEqual(d.importance_delta, 0.0)
        # off-schema -> novel with importance boost
        d = decide([0.3, 0.1])
        self.assertEqual(d.reason, "novel")
        self.assertGreater(d.importance_delta, 0.0)
        # empty prior -> everything is surprising
        self.assertEqual(gate.evaluate(fact, []).reason, "novel")

    def test_volatile_conflict_bypasses_gate_band(self):
        eng = fresh_engine()
        eng.add("我叫李雷，住在杭州", user_id="g3")
        r = eng.add("我搬到上海了", user_id="g3")
        self.assertEqual(len(r.updated), 1)
        mem = [m for m in eng.get_all("g3") if "上海" in m.text][0]
        self.assertEqual(mem.metadata.get("gate"), "volatile-update")
        eng.close()

    def test_gate_can_be_disabled(self):
        eng = fresh_engine(gate_enabled=False)
        eng.add("我叫李雷", user_id="g4")
        r = eng.add("我叫李雷", user_id="g4")
        self.assertEqual(r.rejected, [])
        self.assertGreaterEqual(r.noop, 1)  # consolidation handles the dup
        self.assertEqual(len(eng.get_all("g4")), 1)
        eng.close()


class TestBackgroundConsolidation(unittest.TestCase):
    """LangMem-style reflection: cluster -> distill -> supersede."""

    def _seed_cluster(self, eng, user):
        eng.add("用户喜欢吃川菜，最爱水煮鱼", user_id=user, infer=False)
        eng.add("用户喜欢吃川菜，最爱麻婆豆腐", user_id=user, infer=False)
        eng.add("用户喜欢吃川菜，最爱回锅肉", user_id=user, infer=False)
        eng.add("用户在学 Rust 和 WASM", user_id=user, infer=False)

    def test_consolidation_merges_cluster_and_keeps_audit(self):
        eng = fresh_engine()
        self._seed_cluster(eng, "c1")
        report = eng.consolidate_background(user_id="c1")
        self.assertEqual(report["clusters"], 1)
        self.assertEqual(report["members_absorbed"], 3)
        summary = eng.get(report["summaries"][0]["id"])
        self.assertIn("综合记忆", summary.text)
        self.assertIn("川菜", summary.text)
        self.assertTrue(summary.metadata["consolidated"])
        self.assertEqual(summary.source, "consolidation")
        # bank shrank: 1 unrelated + 1 summary, originals superseded
        mems = eng.get_all("c1")
        self.assertEqual(len(mems), 2)
        self.assertTrue(any("Rust" in m.text for m in mems))
        self.assertFalse(any("水煮鱼" in m.text and not m.metadata.get("consolidated")
                             for m in mems))
        # superseded chain + CONSOLIDATE audit events
        for mid in report["summaries"][0]["members"]:
            member = eng.get(mid)
            self.assertIsNotNone(member.invalid_at)
            self.assertEqual(member.superseded_by, summary.id)
            hist = eng.get_history(mid)
            self.assertTrue(any(e["event"] == "CONSOLIDATE" for e in hist))
            self.assertEqual(hist[0]["new_value"], summary.id)
        # summary carries the strongest member's importance (promoted to core)
        self.assertGreaterEqual(summary.importance, 0.4)
        eng.close()

    def test_summary_stays_retrievable(self):
        eng = fresh_engine()
        self._seed_cluster(eng, "c2")
        eng.consolidate_background(user_id="c2")
        hits = eng.search("用户喜欢吃什么菜", user_id="c2")
        self.assertTrue(hits)
        self.assertIn("综合记忆", hits[0].memory.text)
        block = eng.render_context(user_id="c2")
        self.assertIn("综合记忆", block)
        eng.close()

    def test_small_clusters_untouched_and_no_remerge(self):
        eng = fresh_engine()
        eng.add("用户喜欢吃粤菜早茶", user_id="c3", infer=False)
        eng.add("用户喜欢吃粤菜点心", user_id="c3", infer=False)
        self._seed_cluster(eng, "c4")
        eng.consolidate_background(user_id="c3")
        self.assertEqual(len(eng.get_all("c3")), 2)  # below min_cluster -> untouched
        # first pass merges the c4 cluster; second pass must not re-merge
        # the summary (one abstraction level per run)
        self.assertEqual(eng.consolidate_background(user_id="c4")["clusters"], 1)
        report2 = eng.consolidate_background(user_id="c4")
        self.assertEqual(report2["clusters"], 0)
        eng.close()


class TestBugfixRegression(unittest.TestCase):
    """Regression tests for the 2026-08-30 full-sweep bug audit."""

    def test_llm_delete_op_removes_index_point(self):
        """Regression: a consolidation DELETE decision soft-deleted the PG row
        but left the Qdrant point searchable — stale points permanently
        occupy semantic_topk candidate slots. Same contract as engine.delete."""
        eng = fresh_engine(gate_enabled=False)
        r = eng.add("我叫李雷，住在杭州，喜欢喝美式咖啡", user_id="delvec", infer=False)
        mid = r.added[0]
        before = eng.vector_store.count()

        # DELETE + a NOOP so the fact is consumed and no replacement is added
        from memtide.types import AddResult, ExtractedFact

        result = AddResult()
        eng._apply_fact(ExtractedFact(text="占位事实", entities=[]), {}, result,
                        [{"op": "DELETE", "id": mid},
                         {"op": "NOOP", "id": "nonexistent"}],
                        [], "delvec", None, None, None, [])
        self.assertEqual(result.deleted, [mid])
        self.assertEqual(eng.vector_store.count(), before - 1,
                         "DELETE op must remove the index point")
        self.assertEqual(eng.search("李雷住在哪", user_id="delvec"), [])
        eng.close()

    def test_reset_wipes_everything(self):
        eng = fresh_engine()
        eng.add("我叫李雷，住在杭州", user_id="r1")
        self.assertGreater(len(eng.get_all("r1")), 0)
        eng.reset()
        self.assertEqual(eng.get_all("r1"), [])
        self.assertEqual(eng.store.stats()["active_memories"], 0)
        eng.close()

    def test_bm25_and_entity_channels_respect_agent_scope(self):
        eng = fresh_engine()
        eng.add("用户专有名词量子柠檬比较特殊", user_id="u", agent_id="agentA", infer=False)
        # agentB sees nothing (semantic + bm25 + entity channels all scoped)
        self.assertEqual(eng.search("量子柠檬", user_id="u", agent_id="agentB"), [])
        hits = eng.search("量子柠檬", user_id="u", agent_id="agentA")
        self.assertTrue(hits)
        eng.close()

    def test_manual_update_syncs_vector_index(self):
        eng = fresh_engine()
        eng.add("我叫李雷", user_id="u")
        mid = eng.get_all("u")[0].id
        # Qdrant upsert is verified through the vector-store contract
        calls = []

        class Recorder:
            def upsert(self, mid, vec, payload):
                calls.append((mid, payload))

            def delete(self, mid):
                calls.append(("DEL", mid))

            def clear(self):
                pass

        eng.vector_store = Recorder()
        eng.update(mid, "用户是产品经理")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], mid)
        self.assertEqual(calls[0][1]["user_id"], "u")
        eng.close()

    def test_delete_removes_from_vector_index(self):
        eng = fresh_engine()
        eng.add("我叫李雷", user_id="u")
        mid = eng.get_all("u")[0].id
        calls = []

        class Recorder:
            def upsert(self, *a):
                pass

            def delete(self, mid):
                calls.append(mid)

            def clear(self):
                pass

        eng.vector_store = Recorder()
        eng.delete(mid)
        self.assertEqual(calls, [mid])
        eng.close()


class TestRestApi(unittest.TestCase):
    """Start the real HTTP server on an ephemeral port and exercise all routes."""

    @classmethod
    def setUpClass(cls):
        import threading

        from memtide.server import serve

        cls.tmp = tempfile.TemporaryDirectory()
        httpd = serve(host="127.0.0.1", port=0, config=cfg())
        cls.port = httpd.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        cls.httpd = httpd

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.tmp.cleanup()

    def _req(self, method, path, body=None):
        import json as _json
        import urllib.request

        data = _json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, _json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, _json.loads(e.read().decode())

    def test_crud_and_search(self):
        import urllib.parse
        # add
        code, res = self._req("POST", "/memories", {"text": "我叫李雷，住在杭州", "user_id": "api"})
        self.assertEqual(code, 201)
        self.assertGreaterEqual(len(res["added"]), 2)
        self.assertIn("gate", res)
        mem_id = res["added"][0]
        # list
        code, mems = self._req("GET", "/memories?user_id=api")
        self.assertEqual(code, 200)
        self.assertTrue(any("李雷" in m["text"] for m in mems))
        # get one
        code, mem = self._req("GET", f"/memories/{mem_id}")
        self.assertEqual(code, 200)
        self.assertEqual(mem["id"], mem_id)
        # search
        code, hits = self._req("POST", "/search", {"query": "用户住哪", "user_id": "api"})
        self.assertEqual(code, 200)
        self.assertTrue(hits)
        self.assertIn("components", hits[0])
        # update
        code, res = self._req("PUT", f"/memories/{mem_id}", {"text": "用户是产品经理"})
        self.assertEqual((code, res["updated"]), (200, True))
        # history shows UPDATE
        code, hist = self._req("GET", f"/history?memory_id={mem_id}")
        self.assertTrue(any(e["event"] == "UPDATE" for e in hist))
        # delete
        code, res = self._req("DELETE", f"/memories/{mem_id}")
        self.assertEqual(res["deleted"], True)
        code, _ = self._req("GET", f"/memories/{mem_id}")
        self.assertEqual(code, 404)
        # stats + context
        code, stats = self._req("GET", "/stats")
        self.assertEqual(code, 200)
        self.assertIn("active_memories", stats)
        code, ctx = self._req("GET", "/context?" + urllib.parse.urlencode(
            {"user_id": "api", "query": "用户住哪"}))
        self.assertEqual(code, 200)
        self.assertIn("## Memory", ctx["context"])

    def test_bad_requests(self):
        code, res = self._req("POST", "/memories", {})
        self.assertEqual(code, 400)
        code, _ = self._req("GET", "/memories/nonexistent-id")
        self.assertEqual(code, 404)
        code, _ = self._req("POST", "/search", {})
        self.assertEqual(code, 400)

    def test_malformed_params_get_clean_errors(self):
        # garbage limit used to crash the handler and reset the connection
        code, hits = self._req("POST", "/search", {"query": "test", "limit": "abc"})
        self.assertEqual(code, 200)  # falls back to default limit
        code, _ = self._req("POST", "/memories", {"text": "x", "metadata": ["not", "a", "dict"]})
        self.assertEqual(code, 400)
        code, res = self._req("GET", "/stats?limit=zzz")
        self.assertEqual(code, 200)

    def test_rest_timestamp_roundtrip(self):
        import json as _json

        code, r = self._req("POST", "/memories",
                            {"text": "2024 年的历史对话", "user_id": "tsr",
                             "infer": False, "timestamp": "2024-03-01T09:00:00+00:00"})
        self.assertEqual(code, 201)
        mems = self._req("GET", "/memories?user_id=tsr&include_invalid=true")[1]
        self.assertEqual(mems[0]["created_at"], "2024-03-01T09:00:00+00:00")
        # garbage timestamp -> clean 400
        code, res = self._req("POST", "/memories",
                              {"text": "x", "user_id": "tsr", "timestamp": "很久以前"})  # noqa: F841
        self.assertEqual(code, 400)
        self.assertIn("ISO-8601", res["error"])

    def test_reset_endpoint(self):
        self._req("POST", "/memories", {"text": "重置前记忆", "user_id": "rz"})
        code, res = self._req("POST", "/reset")
        self.assertEqual(code, 400)
        self.assertIn("confirm", res["error"])
        # explicit confirmation is required
        code, res = self._req("POST", "/reset", {"confirm": "RESET"})
        self.assertEqual(code, 200)
        code, mems = self._req("GET", "/memories?user_id=rz")
        self.assertEqual(mems, [])


class TestCallerTimestamp(unittest.TestCase):
    """Historical imports: caller-supplied event time, not ingestion time."""

    def test_timestamp_stamps_memory_and_audit(self):
        eng = fresh_engine()
        r = eng.add("我叫李雷，住在杭州", user_id="ts",
                    timestamp="2024-06-01T10:30:00+00:00")
        self.assertGreaterEqual(len(r.added), 2)
        mem = eng.get(r.added[0])
        self.assertEqual(mem.created_at, "2024-06-01T10:30:00+00:00")
        self.assertEqual(mem.valid_at, "2024-06-01T10:30:00+00:00")
        hist = eng.get_history(r.added[0])
        self.assertEqual(hist[-1]["created_at"], "2024-06-01T10:30:00+00:00")
        eng.close()

    def test_invalid_timestamp_rejected(self):
        eng = fresh_engine()
        with self.assertRaises(ValueError):
            eng.add("我叫李雷", user_id="ts", timestamp="去年夏天")
        eng.close()

    def test_naive_timestamp_normalized_to_utc(self):
        # timezone-less imports are read as UTC so ordering stays consistent
        # when a dataset mixes naive and aware timestamps (issue #5)
        eng = fresh_engine()
        eng.add("我叫王一，住在苏州", user_id="tsn", infer=False,
                timestamp="2024-06-01T10:30:00")
        mems = eng.get_all("tsn")
        self.assertTrue(mems)
        self.assertTrue(all(m.created_at.endswith("+00:00") for m in mems),
                        {m.created_at for m in mems})
        eng.close()

    def test_mixed_naive_and_aware_timestamps_sort_consistently(self):
        eng = fresh_engine()
        eng.add("我叫王一，住在苏州", user_id="tsx", infer=False,
                timestamp="2024-06-01T18:30:00")            # naive (read as UTC)
        eng.add("用户搬到了南京", user_id="tsx", infer=False,
                timestamp="2024-06-02T02:00:00+08:00")       # aware → 18:00Z, older
        mems = [m.created_at for m in eng.get_all("tsx")]
        # every stored timestamp is UTC-normalized, so lexicographic order
        # equals chronological order even across mixed input offsets
        self.assertEqual(sorted(mems), [
            "2024-06-01T18:00:00+00:00", "2024-06-01T18:30:00+00:00"])
        eng.close()


class TestWriteTransactionAtomicity(unittest.TestCase):
    """A crash mid-write must not leave a half-committed state: memories,
    entities and the audit chain roll back together (GitHub issue #1)."""

    def test_insert_rolls_back_on_midwrite_failure(self):
        from memtide.types import Memory

        eng = fresh_engine()
        m = Memory(text="用户喜欢喝乌龙茶", user_id="tx")
        orig = eng.store.log_event_conn
        def boom(*a, **k):
            raise RuntimeError("simulated crash after the memories insert")
        eng.store.log_event_conn = boom
        try:
            with self.assertRaises(RuntimeError):
                eng.store.insert(m, eng.retriever.embed_for_storage(m.text))
        finally:
            eng.store.log_event_conn = orig
        self.assertIsNone(eng.store.get(m.id), "memory row must roll back")
        self.assertEqual(eng.store.history(memory_id=m.id), [])
        eng.close()

    def test_replace_text_rolls_back_on_failure(self):
        eng = fresh_engine()
        r = eng.add("我叫李雷，住在杭州", user_id="tx2")
        mem_id = next(m.id for m in eng.get_all("tx2") if "杭州" in m.text)
        orig = eng.store.log_event_conn
        def boom(*a, **k):
            raise RuntimeError("simulated crash inside replace_text")
        eng.store.log_event_conn = boom
        try:
            with self.assertRaises(RuntimeError):
                eng.store.replace_text(mem_id, "用户住在上海",
                                       ["上海"], eng.retriever.embed_for_storage("用户住在上海"))
        finally:
            eng.store.log_event_conn = orig
        mem = eng.store.get(mem_id)
        self.assertIn("杭州", mem.text, "text update must roll back")
        self.assertTrue(any(e["event"] == "UPDATE" or e["event"] == "ADD"
                            for e in eng.store.history(memory_id=mem_id)))
        eng.close()

    def test_old_memories_rank_lower_via_retention(self):
        eng = fresh_engine()
        eng.add("我喜欢喝美式咖啡", user_id="ts2", timestamp="2024-01-01T00:00:00+00:00")
        eng.add("我喜欢喝美式咖啡", user_id="ts3")  # now
        # a 2024 memory has decayed below the retention floor -> needs re-excitation
        old_ret = eng.search("咖啡", user_id="ts2", include_forgotten=True)[0].components["retention"]
        new_ret = eng.search("咖啡", user_id="ts3")[0].components["retention"]
        self.assertLess(old_ret, 0.5, "a 2024 memory should have decayed by now")
        self.assertGreater(new_ret, 0.99)
        eng.close()


class TestUpdateGuardRegression(unittest.TestCase):
    """PUT on an invalidated memory must not resurrect it (2026-08-30 audit #2)."""

    def test_update_refuses_invalidated_memory(self):
        eng = fresh_engine()
        eng.add("核验用临时记忆", user_id="au", infer=False)
        mid = eng.get_all("au")[0].id
        eng.delete(mid)  # soft delete
        self.assertFalse(eng.update(mid, "被复活的记忆"))
        m = eng.get(mid)
        self.assertIsNotNone(m.invalid_at, "invalid_at must survive an update attempt")
        # still hidden from search and default listings
        self.assertEqual([x for x in eng.get_all("au")], [])
        eng.close()


class TestShortQueryRegression(unittest.TestCase):
    """2-char CJK queries work natively via the ngram BM25 tokenizer."""

    def test_short_cjk_query(self):
        eng = fresh_engine()
        eng.add("我叫李雷，住在杭州，喜欢喝美式咖啡", user_id="q")
        hits = eng.search("咖啡", user_id="q")
        self.assertTrue(hits, "short CJK query must not crash and must hit")
        self.assertIn("咖啡", hits[0].memory.text)
        eng.close()

    def test_render_context_with_short_query(self):
        eng = fresh_engine()
        eng.add("我叫李雷，住在杭州，喜欢喝美式咖啡", user_id="q")
        block = eng.render_context(user_id="q", query="咖啡")
        self.assertIn("## Memory", block)
        self.assertIn("## Relevant to current query", block)
        eng.close()


class TestStaticUI(unittest.TestCase):
    """UI hosting + include_invalid endpoints."""

    @classmethod
    def setUpClass(cls):
        import threading

        from memtide.server import serve

        cls.tmp = tempfile.TemporaryDirectory()
        static_dir = os.path.join(cls.tmp.name, "static")
        os.makedirs(static_dir)
        with open(os.path.join(static_dir, "index.html"), "w") as f:
            f.write("<html><body>memtide-ui-test</body></html>")
        httpd = serve(host="127.0.0.1", port=0, config=cfg())
        os.environ["MEMTIDE_STATIC_DIR"] = static_dir
        cls.port = httpd.server_address[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        cls.httpd = httpd

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.tmp.cleanup()
        os.environ.pop("MEMTIDE_STATIC_DIR", None)

    def _req(self, method, path, body=None):
        import json as _json
        import urllib.request

        data = _json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, resp.read(), dict(resp.headers)
        except urllib.error.HTTPError as e:
            return e.code, e.read(), dict(e.headers)

    def test_root_serves_console_without_landing(self):
        """No landing.html in this static dir -> / falls back to the console."""
        code, body, headers = self._req("GET", "/")
        self.assertEqual(code, 200)
        self.assertIn("memtide-ui-test", body.decode())
        self.assertIn("text/html", headers.get("Content-Type", ""))

    def test_console_and_legacy_alias(self):
        code, body, _ = self._req("GET", "/console")
        self.assertEqual(code, 200)
        self.assertIn("memtide-ui-test", body.decode())
        code, body, _ = self._req("GET", "/ui/")
        self.assertEqual(code, 200)
        self.assertIn("memtide-ui-test", body.decode())
        # console SPA fallback
        code, body, _ = self._req("GET", "/console/some/route")
        self.assertEqual(code, 200)
        self.assertIn("memtide-ui-test", body.decode())

    def test_landing_served_when_present(self):
        landing = os.path.join(self.tmp.name, "static", "landing.html")
        with open(landing, "w") as f:
            f.write("<html><body>memtide-official-site</body></html>")
        code, body, _ = self._req("GET", "/")
        self.assertEqual(code, 200)
        self.assertIn("memtide-official-site", body.decode())
        # unknown non-API paths fall back to the landing page
        code, body, _ = self._req("GET", "/whatever")
        self.assertEqual(code, 200)
        self.assertIn("memtide-official-site", body.decode())
        # unknown API paths still answer JSON 404
        code, body, _ = self._req("GET", "/memories/does-not-exist")
        self.assertEqual(code, 404)
        os.remove(landing)

    def test_include_invalid_list_and_detail(self):
        import json as _json

        code, body, _ = self._req("POST", "/memories",
                                  {"text": "将被删除的记忆", "user_id": "iv", "infer": False})
        mem_id = _json.loads(body)["added"][0]
        self._req("DELETE", f"/memories/{mem_id}")
        # default: gone from list and detail 404
        code, body, _ = self._req("GET", "/memories?user_id=iv")
        self.assertEqual(_json.loads(body), [])
        code, _, _ = self._req("GET", f"/memories/{mem_id}")
        self.assertEqual(code, 404)
        # include_invalid: present with invalid flag
        code, body, _ = self._req("GET", "/memories?user_id=iv&include_invalid=true")
        mems = _json.loads(body)
        self.assertEqual(len(mems), 1)
        self.assertIsNotNone(mems[0]["invalid_at"])
        code, body, _ = self._req("GET", f"/memories/{mem_id}?include_invalid=true")
        self.assertEqual(code, 200)

    def test_rebuild_endpoint(self):
        import json as _json

        self._req("POST", "/memories", {"text": "重建测试记忆", "user_id": "rb", "infer": False})
        code, body, _ = self._req("POST", "/rebuild")
        self.assertEqual(code, 200)
        self.assertGreaterEqual(_json.loads(body)["reindexed"], 1)


class TestOpenAIPath(unittest.TestCase):
    """The real-LLM path is exercised with a fake HTTP server (no network)."""

    def test_openai_llm_and_embeddings(self):
        import json as _json
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import threading

        calls = []

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append(self.path)
                if self.path.endswith("/embeddings"):
                    text = body["input"]
                    dim = 8
                    vec = [float((len(text) + i) % 3) for i in range(dim)]
                    n = sum(v * v for v in vec) ** 0.5 or 1
                    vec = [v / n for v in vec]
                    out = {"data": [{"embedding": vec}]}
                else:
                    out = {"choices": [{"message": {"content": _json.dumps({
                        "facts": [{"text": "The user's name is Bob", "type": "fact",
                                   "importance": 0.9, "entities": ["Bob"]}]})}}]}
                resp = _json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), H)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            cfg = MemoryConfig(
                llm_backend="openai",
                llm_base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                llm_api_key="test-key",
                embedding_backend="openai",
                pg_dsn=new_schema_dsn()[0],
            )
            eng = MemoryEngine(cfg)
            res = eng.add("Hi, I'm Bob")
            self.assertEqual(res.facts, ["The user's name is Bob"])
            self.assertEqual(len(res.added), 1)
            hits = eng.search("what is the user's name?")
            self.assertTrue(hits)
            self.assertIn("Bob", hits[0].memory.text)
            eng.close()
        finally:
            srv.shutdown()


class TestMultimodal(unittest.TestCase):
    """Multimodal ingestion: media parts -> description text + attachments."""

    PNG = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==")

    def setUp(self):
        import tempfile as _tf

        self.tmp = _tf.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.media_dir = os.path.join(self.tmp.name, "media")
        self.eng = fresh_engine(media_dir=self.media_dir)

    def _data_url(self, png=None):
        png = png or self.PNG
        return "data:image/png;base64," + base64.b64encode(png).decode()

    def _add_image(self, eng=None, caption="A handwritten note that says my name is Wang Lei"):
        from unittest import mock

        from memtide import multimodal

        eng = eng or self.eng
        with mock.patch.object(multimodal, "describe_image", return_value=caption):
            return eng.add(
                [{"role": "user", "content": [
                    {"type": "text", "text": "看看这张图"},
                    {"type": "image_url", "image_url": {"url": self._data_url()}},
                ]}],
                user_id="mm")

    def test_image_part_yields_facts_and_attachment(self):
        res = self._add_image()
        self.assertTrue(res.attachments)
        att = res.attachments[0]
        self.assertEqual(att["kind"], "image")
        self.assertEqual(att["mime"], "image/png")
        self.assertIn("Wang Lei", att["description"])
        # content-addressed file exists on disk
        path = os.path.join(self.media_dir, att["source"])
        self.assertTrue(os.path.isfile(path))
        with open(path, "rb") as f:
            self.assertEqual(f.read(), self.PNG)
        # facts extracted from the description carry the attachment + modality
        mems = self.eng.get_all("mm")
        linked = [m for m in mems if "Wang Lei" in m.text]
        self.assertTrue(linked)
        self.assertTrue(all(m.attachments for m in linked))
        self.assertEqual(linked[0].metadata.get("modality"), "image")
        # text query retrieves the image-derived memory (cross-modal)
        hits = self.eng.search("name Wang Lei", user_id="mm")
        self.assertTrue(hits)
        self.assertIn("Wang Lei", hits[0].memory.text)
        # attachments must survive serialization (search + get)
        self.assertTrue(hits[0].to_dict()["attachments"])
        self.assertTrue(self.eng.get(hits[0].memory.id).to_dict()["attachments"])

    def test_same_bytes_dedup_on_disk(self):
        self._add_image()
        self._add_image()
        self.assertEqual(len(os.listdir(self.media_dir)), 1)

    def test_image_only_call_keeps_description(self):
        # nothing extractable from the caption -> description itself is stored
        from unittest import mock

        from memtide import multimodal

        with mock.patch.object(multimodal, "describe_image",
                               return_value="A photo of a mountain lake at sunrise"):
            res = self.eng.add([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": self._data_url()}}]}],
                user_id="img-only")
        self.assertEqual(len(res.added), 1)
        mem = self.eng.get(res.added[0])
        self.assertIn("mountain lake", mem.text)
        self.assertEqual(mem.attachments[0]["kind"], "image")

    def test_infer_false_links_attachment(self):
        from unittest import mock

        from memtide import multimodal

        with mock.patch.object(multimodal, "describe_image", return_value="caption X"):
            res = self.eng.add(
                [{"role": "user", "content": [
                    {"type": "text", "text": "看这张"},
                    {"type": "image_url", "image_url": {"url": self._data_url()}},
                ]}],
                user_id="raw", infer=False)
        mem = self.eng.get(res.added[0])
        self.assertIn("caption X", mem.text)
        self.assertEqual(len(mem.attachments), 1)

    def test_vision_failure_degrades_gracefully(self):
        # endpoint unreachable -> media still stored, no description, no crash
        from unittest import mock

        from memtide import multimodal

        with mock.patch.object(multimodal, "describe_image", return_value=None):
            res = self.eng.add([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": self._data_url()}}]}],
                user_id="nofail")
        self.assertTrue(res.attachments)
        self.assertIsNone(res.attachments[0]["description"])

    def test_describe_image_handles_unreachable_endpoint(self):
        from memtide import multimodal

        c = cfg(media_dir=self.media_dir, llm_backend="openai",
                llm_base_url="http://127.0.0.1:1/v1", llm_api_key="k")
        # connection refused -> None, not an exception
        self.assertIsNone(multimodal.describe_image(self._data_url(), c))

    def test_multimodal_disabled_ignores_media(self):
        eng = fresh_engine(media_dir=self.media_dir, multimodal_enabled=False)
        res = eng.add([{"role": "user", "content": [
            {"type": "text", "text": "我叫李雷"},
            {"type": "image_url", "image_url": {"url": self._data_url()}},
        ]}], user_id="off")
        self.assertEqual(res.attachments, [])
        self.assertEqual(os.listdir(self.media_dir) if os.path.isdir(self.media_dir) else [], [])
        self.assertTrue(any("李雷" in t for t in res.facts))

    def test_oversized_media_rejected(self):
        from memtide import multimodal

        c = cfg(media_dir=self.media_dir, max_media_bytes=10)
        part = {"type": "image_url", "image_url": {"url": self._data_url()}}
        with self.assertRaises(ValueError):
            multimodal.process_part(part, c)

    def test_local_path_disabled_by_default(self):
        """SECURITY: local-path ingest is opt-in — otherwise a REST deployment
        would let any caller read arbitrary local files via /media/."""
        from memtide import multimodal

        real_png = os.path.join(self.tmp.name, "real.png")
        with open(real_png, "wb") as f:
            f.write(self.PNG)
        c = cfg(media_dir=self.media_dir)
        with self.assertRaises(ValueError):
            multimodal.process_part({"type": "file", "path": "/etc/passwd"}, c)
        with self.assertRaises(ValueError):
            multimodal.process_part({"type": "image", "path": real_png}, c)
        # opt-in for trusted embedders: works, and missing files are clean 400s
        c2 = cfg(media_dir=self.media_dir, media_allow_paths=True)
        att = multimodal.process_part({"type": "image", "path": real_png}, c2)
        self.assertEqual(att.kind, "image")
        with open(os.path.join(self.media_dir, att.source), "rb") as f:
            self.assertEqual(f.read(), self.PNG)
        with self.assertRaises(ValueError):
            multimodal.process_part({"type": "image", "path": "/nonexistent/x.png"}, c2)

    def test_rest_media_endpoint(self):
        import json as _json
        import threading
        import urllib.request

        from memtide.server import serve

        httpd = serve(host="127.0.0.1", port=0,
                      config=cfg(media_dir=self.media_dir))
        port = httpd.server_address[1]
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        self.addCleanup(httpd.shutdown)

        def req(method, path, body=None):
            data = _json.dumps(body).encode() if body is not None else None
            r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                       method=method,
                                       headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(r, timeout=5) as resp:
                    return resp.status, resp.read(), resp.headers
            except urllib.error.HTTPError as e:
                return e.code, e.read(), e.headers

        from unittest import mock

        from memtide import multimodal

        with mock.patch.object(multimodal, "describe_image",
                               return_value="A note that says my name is Wang Lei"):
            code, body, _ = req("POST", "/memories", {
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": self._data_url()}}]}],
                "user_id": "rest"})
            self.assertEqual(code, 201)
            atts = _json.loads(body.decode()).get("attachments", [])
            self.assertTrue(atts)
            # serve the asset back
            code, raw, headers = req("GET", f"/media/{atts[0]['sha256']}")
            self.assertEqual(code, 200)
            self.assertEqual(raw, self.PNG)
            self.assertEqual(headers["Content-Type"], "image/png")
            # path traversal / garbage -> 404
            code, _, _ = req("GET", "/media/zzzz")
            self.assertEqual(code, 404)
            code, _, _ = req("GET", f"/media/{'a' * 64}")
            self.assertEqual(code, 404)


class TestOptimizations(unittest.TestCase):
    """Optimization pack: write-path perf, retrieval quality, gate scoping,
    lifecycle ops, REST auth/export/import/background."""

    # ---- A1: single-query similarity scan -----------------------------------
    def test_similarity_scan_is_not_n_plus_one(self):
        from memtide.types import Memory

        eng = fresh_engine()
        for i in range(30):  # 30 existing memories
            m = Memory(text=f"记忆条目编号{i}号", user_id="perf")
            eng.store.insert(m, eng.retriever.embed_for_storage(m.text))
        calls = {"n": 0}
        orig = eng.store.get_embedding

        def counting(mid):
            calls["n"] += 1
            return orig(mid)

        eng.store.get_embedding = counting
        eng.add("我叫李雷，住在杭州，喜欢咖啡", user_id="perf")
        # old code: one get_embedding per memory PER FACT (~90+); the scan now
        # uses one all_embeddings query and only hydrates top candidates
        self.assertLess(calls["n"], 20, calls["n"])

    # ---- A3: cached embedder + real batch API call ---------------------------
    def test_cached_embedder_hits(self):
        from memtide.embeddings import CachedEmbedder
        from tests.fake_openai import FakeEmbedder

        inner = FakeEmbedder(64)
        calls = {"n": 0}
        orig = inner.embed

        def counting(t):
            calls["n"] += 1
            return orig(t)

        inner.embed = counting
        ce = CachedEmbedder(inner)
        ce.embed("hello"); ce.embed("hello")
        self.assertEqual(calls["n"], 1)
        ce.embed_batch(["a", "hello", "b"])
        self.assertEqual(calls["n"], 3)  # "hello" cached, a/b computed once

    def test_openai_embedder_batches_into_one_request(self):
        import json as _json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        from memtide.embeddings import OpenAIEmbedder

        calls = {"n": 0}

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls["n"] += 1
                inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
                vecs = [[float((len(t) + i) % 3) for i in range(8)] for t in inputs]
                out = {"data": [{"index": i, "embedding": v} for i, v in enumerate(vecs)]}
                resp = _json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            def log_message(self, *a):
                pass

        srv = HTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            emb = OpenAIEmbedder(f"http://127.0.0.1:{srv.server_port}/v1", "k", "m")
            vecs = emb.embed_batch(["x", "y", "x"])  # 2 uncached -> ONE request
            self.assertEqual(calls["n"], 1)
            self.assertEqual(len(vecs), 3)
            emb.embed("x")  # cached now
            self.assertEqual(calls["n"], 1)
        finally:
            srv.shutdown()

    # ---- C1: slot-scoped gate prior ------------------------------------------
    def test_gate_slot_scoped_prior(self):
        from memtide.gating import PredictiveGate
        from memtide.types import ExtractedFact, Memory

        mem_role = Memory(text="用户是工程师", user_id="x", metadata={"slot": "role"})
        fact = ExtractedFact(text="用户住在杭州", memory_type="fact", slot="location")
        gate = PredictiveGate(cfg())
        # same surface similarity 0.9, but the only predictor is a DIFFERENT
        # slot -> with slot scoping this is NOT redundant
        d = gate.evaluate(fact, [(0.9, mem_role)])
        self.assertNotEqual(d.reason, "redundant")
        # with scoping disabled the old global-max behavior returns
        gate2 = PredictiveGate(cfg(gate_slot_scoped=False))
        d2 = gate2.evaluate(fact, [(0.9, mem_role)])
        self.assertEqual(d2.reason, "redundant")

    # ---- B1: type/slot filters ------------------------------------------------
    def test_search_type_and_slot_filters(self):
        from memtide.types import Memory

        eng = fresh_engine()
        eng.add("我叫李雷，喜欢喝咖啡", user_id="f")
        m = Memory(text="用户住在苏州", user_id="f", metadata={"slot": "location"})
        eng.store.insert(m, eng.retriever.embed_for_storage(m.text))
        prefs = eng.search("咖啡", user_id="f", memory_type="preference")
        self.assertTrue(prefs)
        self.assertTrue(all(h.memory.memory_type == "preference" for h in prefs))
        loc = eng.search("用户住哪", user_id="f", slot="location")
        self.assertTrue(loc)
        self.assertEqual(loc[0].memory.metadata.get("slot"), "location")
        none = eng.search("咖啡", user_id="f", memory_type="procedural")
        self.assertEqual(none, [])

    # ---- B4: MMR diversity -----------------------------------------------------
    def test_mmr_prefers_diverse_results(self):
        from memtide.types import Memory

        eng = fresh_engine()
        uid = "mmr"
        # two near-identical rows (inserted raw so consolidation can't merge)
        eng.add("用户喜欢喝美式咖啡配燕麦奶", user_id=uid, infer=False)
        twin = Memory(text="用户喜欢喝美式咖啡加燕麦奶", user_id=uid)
        twin_emb = eng.retriever.embed_for_storage(twin.text)
        eng.store.insert(twin, twin_emb)
        eng._index_upsert(twin, twin_emb)  # raw insert must keep Qdrant in sync
        eng.add("用户在杭州一家创业公司做后端开发", user_id=uid, infer=False)
        eng.cfg.mmr_lambda = 0.0
        plain = [h.memory.text for h in eng.search("用户喜欢喝什么咖啡", user_id=uid, limit=2)]
        self.assertEqual(len(plain), 2)
        eng.cfg.mmr_lambda = 0.2
        diverse = [h.memory.text for h in eng.search("用户喜欢喝什么咖啡", user_id=uid, limit=2)]
        self.assertEqual(len(diverse), 2)
        # MMR should make room for a less-similar memory when the near-duplicate
        # pair dominates the plain top-k (Qdrant may return either twin first).
        if all("咖啡" in t for t in plain):
            self.assertTrue(any("后端开发" in t for t in diverse), diverse)

    # ---- B2/B3: reranker + query expansion via a fake OpenAI-style server ----
    def _fake_llm_server(self, handler_fn):
        import json as _json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        srv = HTTPServer(("127.0.0.1", 0), handler_fn)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv

    def test_query_expansion_retrieves_via_variant(self):
        import json as _json
        from http.server import BaseHTTPRequestHandler

        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                system = body["messages"][0]["content"]
                if "memory extraction engine" in system:
                    content = _json.dumps({"facts": [
                        {"text": "用户的名字是李雷", "type": "fact", "importance": 0.9,
                         "entities": ["李雷"], "slot": "name"},
                        {"text": "用户住在杭州", "type": "fact", "importance": 0.9,
                         "entities": ["杭州"], "slot": "location"},
                    ]})
                elif "optimize retrieval" in system:
                    outer.assertIn("where does the user live", body["messages"][1]["content"])
                    content = _json.dumps({"variants": ["the user lives in Hangzhou",
                                                        "用户住在杭州"]})
                elif "memory consolidation" in system:
                    content = _json.dumps({"operations": []})
                else:
                    content = "{}"
                resp = _json.dumps({"choices": [{"message": {"content": content}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            def log_message(self, *a):
                pass

        srv = self._fake_llm_server(H)
        try:
            eng = fresh_engine(llm_backend="openai",
                                   llm_base_url=f"http://127.0.0.1:{srv.server_port}/v1",
                                   llm_api_key="k", query_expansion=True)
            eng.add([{"role": "user", "content": "我叫李雷，住在杭州"}], user_id="qe")
            # English query vs a Chinese memory: expansion's Chinese variant
            # must bridge the lexical gap (without it, hashed-embedder recall fails)
            eng.cfg.query_expansion = False
            hits_plain = eng.search("where does the user live", user_id="qe")
            eng.cfg.query_expansion = True
            hits_exp = eng.search("where does the user live", user_id="qe")
            self.assertTrue(any("杭州" in h.memory.text for h in hits_exp),
                            [h.memory.text for h in hits_exp])
        finally:
            srv.shutdown()

    def test_http_reranker_reorders(self):
        import json as _json
        from http.server import BaseHTTPRequestHandler

        state = {"calls": 0}

        class R(BaseHTTPRequestHandler):
            def do_POST(self):
                state["calls"] += 1
                body = _json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                # rank by shortest text last -> reversed document order
                docs = body["documents"]
                order = sorted(range(len(docs)), key=lambda i: -len(docs[i]))
                out = {"results": [{"index": i, "relevance_score": 1.0 - rank / 10}
                                   for rank, i in enumerate(order)]}
                resp = _json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

            def log_message(self, *a):
                pass

        rsrv = self._fake_llm_server(R)
        try:
            eng = fresh_engine(rerank_backend="http",
                                   rerank_base_url=f"http://127.0.0.1:{rsrv.server_port}",
                                   rerank_model="test-rerank")
            eng.add("我喜欢美式咖啡", user_id="rr", infer=False)
            eng.add("我每天早上去公园散步然后买一杯美式咖啡这是一种持续了很多年的习惯", user_id="rr", infer=False)
            hits = eng.search("美式咖啡", user_id="rr", limit=5)
            self.assertGreaterEqual(len(hits), 2)
            self.assertEqual(state["calls"], 1)
            self.assertIn("rerank", hits[0].components)
            # the long habit sentence gets the top rerank score under the fake
            self.assertIn("公园", hits[0].memory.text)
        finally:
            rsrv.shutdown()

    # ---- D-group: lifecycle ------------------------------------------------
    def test_compaction_merges_near_duplicates(self):
        from memtide.types import Memory

        eng = fresh_engine()
        text = "我叫李雷，住在杭州，喜欢喝美式咖啡"
        for _ in range(2):  # double-import artifact: identical rows
            m = Memory(text=text, user_id="cp")
            eng.store.insert(m, eng.retriever.embed_for_storage(text))
        eng.add("我在学 Rust", user_id="cp", infer=False)
        rep = eng.compact("cp")
        self.assertEqual(rep["absorbed"], 1)
        self.assertEqual(len(rep["kept"]), 1)
        self.assertEqual(len(eng.get_all("cp")), 2)
        # audit chain kept the absorbed member
        hist = [e for e in eng.get_history(rep["kept"][0])]
        self.assertTrue(hist)

    def test_media_gc_removes_orphans(self):
        eng = MemoryEngine(cfg(media_dir=os.path.join(self.tmpdir(), "media")))
        os.makedirs(eng.cfg.media_dir, exist_ok=True)
        orphan = "0" * 64 + ".png"
        with open(os.path.join(eng.cfg.media_dir, orphan), "wb") as f:
            f.write(b"x")
        r = eng.media_gc()  # dry run by default
        self.assertEqual(r["orphan"], 1)
        self.assertEqual(r["removed"], [])
        self.assertTrue(os.path.exists(os.path.join(eng.cfg.media_dir, orphan)))
        r = eng.media_gc(delete=True)
        self.assertEqual(r["removed"], [orphan])
        self.assertFalse(os.path.exists(os.path.join(eng.cfg.media_dir, orphan)))

    def test_consolidation_memories_fade_slower(self):
        from datetime import datetime, timedelta, timezone

        from memtide.decay import retention
        from memtide.types import Memory

        old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        plain = Memory(text="细节", user_id="d", source="conversation", created_at=old,
                       updated_at=old, valid_at=old)
        summary = Memory(text="概括", user_id="d", source="consolidation", created_at=old,
                         updated_at=old, valid_at=old)
        r_plain = retention(plain, 45, 0.4)
        r_summary = retention(summary, 45, 0.4, consolidation_mult=3.0)
        self.assertGreater(r_summary, r_plain * 1.5)

    def test_auto_reflect_scheduler_runs(self):
        # gate off: these texts sit right on the redundant boundary under the
        # the deterministic test embedder sits on the gate boundary; test the scheduler, not the gate
        eng = fresh_engine(gate_enabled=False)
        for t in ["我正在学习 Rust 之一", "我正在学习 Rust 之二", "我正在学习 Rust 之三"]:
            eng.add(t, user_id="sched", infer=False)
        before = len(eng.get_all("sched"))
        eng.enable_auto_reflect(interval_seconds=0.05)
        self.assertTrue(eng.auto_reflect_active)
        deadline = time.time() + 8
        while time.time() < deadline:
            if len(eng.get_all("sched")) < before:
                break
            time.sleep(0.05)
        eng.disable_auto_reflect()
        self.assertFalse(eng.auto_reflect_active)
        self.assertLess(len(eng.get_all("sched")), before, "reflection should absorb a cluster")

    def test_background_add_returns_future(self):
        eng = fresh_engine()
        fut = eng.add_background("我叫王五", user_id="bg")
        res = fut.result(timeout=5)
        self.assertTrue(res.added)
        self.assertTrue(eng.get(res.added[0]))
        eng.close()

    def test_export_import_roundtrip(self):
        from memtide.types import Memory

        eng = fresh_engine()
        eng.add("我叫李雷，住在杭州", user_id="ex", infer=False)
        m = Memory(text="已删除的记忆", user_id="ex")
        eng.store.insert(m, eng.retriever.embed_for_storage(m.text))
        eng.delete(m.id)  # invalidated -> must survive export too
        dump = os.path.join(self.tmpdir(), "dump.jsonl")
        eng.export_jsonl(dump, user_id="ex")
        self.assertTrue(os.path.getsize(dump) > 0)

        eng2 = fresh_engine()
        with open(dump, encoding="utf-8") as f:
            stats = eng2.import_jsonl(f.read().splitlines())
        self.assertEqual(stats["imported"], 2, stats)
        self.assertTrue(eng2.search("用户住哪", user_id="ex"))
        # invalidated memory restored with its audit state
        rows = eng2.store.all_rows(user_id="ex", include_invalid=True)
        self.assertTrue(any(r.invalid_at for r in rows))
        # conflict on import: skip by default
        stats2 = eng2.import_jsonl(open(dump, encoding="utf-8").read().splitlines())
        self.assertEqual(stats2["skipped"], 2)
        eng.close(); eng2.close()

    # ---- REST: auth + export/import endpoints --------------------------------
    def test_rest_auth_export_import(self):
        import json as _json
        import threading
        import urllib.request

        from memtide.server import serve

        tmpdir = self.tmpdir()
        httpd = serve(host="127.0.0.1", port=0,
                      config=cfg(api_key="secret-key"))
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.shutdown)

        def req(method, path, body=None, key="secret-key", raw=False):
            data = _json.dumps(body).encode() if body is not None else None
            r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data,
                                       method=method,
                                       headers={"Content-Type": "application/json",
                                                "X-API-Key": key})
            try:
                with urllib.request.urlopen(r, timeout=5) as resp:
                    payload = resp.read()
                    return resp.status, (payload if raw else _json.loads(payload))
            except urllib.error.HTTPError as e:
                return e.code, _json.loads(e.read().decode())

        code, _ = req("GET", "/stats", key="")
        self.assertEqual(code, 401)
        code, _ = req("GET", "/stats")
        self.assertEqual(code, 200)
        code, res = req("POST", "/memories", {"text": "我叫李雷", "user_id": "auth"})
        self.assertEqual(code, 201)
        code, raw = req("GET", "/export?user_id=auth", raw=True)
        self.assertEqual(code, 200)
        lines = [ln for ln in raw.decode().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1)
        code, imp = req("POST", "/import", {"lines": [_json.loads(lines[0])]})
        self.assertEqual((code, imp["skipped"]), (200, 1))
        code, rep = req("POST", "/compact", {"user_id": "auth"})
        self.assertEqual(code, 200)
        code, rep = req("POST", "/media/gc", {})
        self.assertEqual(code, 200)

    def test_calibration_script_runs(self):
        import subprocess

        env = {**os.environ, "LLM_BASE_URL": FAKE_URL, "LLM_API_KEY": "test", "EMBEDDING_BACKEND": "openai", "EMBEDDING_BASE_URL": FAKE_URL}
        r = subprocess.run(
            [sys.executable, os.path.join(os.path.dirname(__file__), "..",
                                          "scripts", "calibrate_gate.py")],
            capture_output=True, text=True, timeout=120, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("gate_redundant_bits", r.stdout)

    def test_concurrent_background_adds_and_reads(self):
        """Engine-level lock: a background-add pool and main-thread reads share
        one PostgreSQL connection; concurrent use must never raise driver errors
        (regression: auto-reflect scheduler raced main-thread polling)."""
        eng = fresh_engine(gate_enabled=False)  # under test: concurrency, not the gate
        futures = [eng.add_background(f"并发写入测试第{i}号记忆", user_id="race",
                                      infer=False)
                   for i in range(16)]
        for _ in range(200):  # hammer reads from the main thread meanwhile
            eng.get_all("race")
            eng.stats()
        # must all resolve without driver errors; consolidation may
        # legitimately merge near-identical texts, so just require one survivor
        results = [f.result(timeout=10) for f in futures]
        self.assertTrue(any(r.added for r in results))
        self.assertTrue(eng.get_all("race", limit=100))
        eng.close()

    def test_pool_does_not_leak_connections(self):
        """Regression: a bounded pool must return every borrowed connection —
        a leak would pile up 'too many clients' until PG refuses new ones
        (GitHub incident: dashboard polling + per-request threads exhausted
        max_connections)."""
        eng = fresh_engine(gate_enabled=False)
        store = eng.store
        baseline = store._pool.qsize()

        # exercise every read/write path that borrows a connection
        r = eng.add("我叫李雷，住在杭州，喜欢喝美式咖啡", user_id="leak")
        eng.search("用户住哪", user_id="leak")
        eng.get_all("leak")
        eng.get_history(r.added[0])
        eng.stats()
        eng.render_context(user_id="leak", query="用户住哪")
        eng.consolidate_background(user_id="leak")
        eng.export_jsonl(user_id="leak", include_embeddings=True)

        # all borrowed connections must be back in the pool
        self.assertGreaterEqual(store._pool.qsize(), baseline,
                                "pool lost connections (leak)")
        eng.close()

    def test_pool_enforces_connection_cap(self):
        """Regression: the cap must count connections ever created, not the
        idle queue. Gating on qsize() was vacuous — the queue is empty at the
        moment of the check, so borrowers just opened new connections and the
        'max_conns' bound was never enforced. Hold every pooled connection and
        the next borrow must block out its grace period, then fail loudly."""
        eng = fresh_engine(gate_enabled=False)
        store = eng.store

        held = []
        try:
            with self.assertRaises(ConnectionLimitError):
                while True:
                    ctx = store._acquire()
                    ctx.__enter__()
                    held.append(ctx)
        finally:
            for ctx in held:
                ctx.__exit__(None, None, None)

        self.assertLessEqual(store._opened, store._max_conns,
                             "cap exceeded: more connections opened than max_conns")
        # every held connection is back; a new borrow is served from the pool
        # without opening anything new
        with store._acquire() as conn:
            conn.execute("SELECT 1")
        self.assertLessEqual(store._opened, store._max_conns)
        eng.close()

    # ---- helper ---------------------------------------------------------------
    def tmpdir(self):
        import tempfile as _tf

        if not hasattr(self, "_tmp"):
            self._tmp = _tf.TemporaryDirectory()
            self.addCleanup(self._tmp.cleanup)
        return self._tmp.name


class TestBatch3RetrievalDecay(unittest.TestCase):
    """Batch-3: entity stopwords, weighted RRF, episodic decay, hl cap."""

    def test_entity_stopwords_filtered(self):
        from memtide.retrieval import _extract_query_entities

        ents = _extract_query_entities("用户现在喜欢什么？告诉我")
        for stop in ("什么", "现在", "喜欢", "告诉"):
            self.assertNotIn(stop, ents)
        # quoted spans still win as high-precision entities
        self.assertIn("ProjX", _extract_query_entities('跟我说说"ProjX"的进展'))

    def test_weighted_rrf_entity_counts_less(self):
        from memtide.retrieval import _rrf

        eq = _rrf([["a", "b"], ["b", "a"]], 60)
        self.assertAlmostEqual(eq["a"], eq["b"])
        w = _rrf([["a", "b"], ["b", "a"]], 60, [1.0, 0.5])
        self.assertGreater(w["a"], w["b"])

    def test_episodic_fades_faster_and_floor_higher(self):
        from datetime import datetime, timedelta, timezone

        from memtide.decay import is_forgotten, retention
        from memtide.types import Memory

        old = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        fact = Memory(text="f", user_id="d", memory_type="fact",
                      created_at=old, updated_at=old, valid_at=old)
        epi = Memory(text="e", user_id="d", memory_type="episodic",
                     created_at=old, updated_at=old, valid_at=old)
        self.assertLess(retention(epi, 45, 0.4), retention(fact, 45, 0.4))
        # episodic hits its higher floor first
        self.assertTrue(is_forgotten(epi, 45, 0.4, 0.02, episodic_floor=0.99))
        self.assertFalse(is_forgotten(fact, 45, 0.4, 0.02, episodic_floor=0.99))

    def test_half_life_capped(self):
        from memtide.decay import effective_half_life
        from memtide.types import Memory

        hot = Memory(text="h", user_id="d", access_count=10 ** 9)
        self.assertLessEqual(effective_half_life(hot, 45, 0.4), 45 * 4.0)

    def test_slot_filter_matches_alias(self):
        from memtide.types import Memory

        eng = fresh_engine()
        m = Memory(text="用户住在苏州", user_id="al", metadata={"slot": "location"})
        eng.store.insert(m, eng.retriever.embed_for_storage(m.text))
        self.assertTrue(eng.search("用户住哪", user_id="al", slot="city"))
        eng.close()

    def test_render_context_preview_does_not_reinforce(self):
        eng = fresh_engine()
        eng.add("我叫李雷，住在杭州", user_id="nr")
        total = lambda: sum(m.access_count for m in eng.get_all("nr", limit=100))
        before = total()
        eng.render_context(user_id="nr", query="用户住哪")
        self.assertEqual(total(), before)
        eng.search("用户住哪", user_id="nr")  # normal search still reinforces
        self.assertGreater(total(), before)
        eng.close()

    def test_age_ignores_access_rejuvenation(self):
        from datetime import datetime, timedelta, timezone

        from memtide.decay import age_days
        from memtide.types import Memory

        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        now = datetime.now(timezone.utc).isoformat()
        m = Memory(text="m", user_id="d", created_at=old, updated_at=old,
                   valid_at=old, last_accessed=now, access_count=5)
        self.assertGreater(age_days(m), 29.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

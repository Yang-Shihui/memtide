"""Live integration tests against REAL endpoints — skipped by default.

Run with real keys:
  MEMTIDE_LIVE=1 \
  MEMTIDE_LLM_BASE_URL=http://.../v1 MEMTIDE_LLM_KEY=sk-... MEMTIDE_LLM_MODEL=GLM-5.3-Flash \
  MEMTIDE_DASHSCOPE_KEY=sk-... \
  python3 -m unittest tests.test_live -v

These hit the network and cost tokens; the regular suite (test_memtide.py)
stays hermetic.
"""

import os
import sys
import unittest
import uuid

LIVE = os.environ.get("MEMTIDE_LIVE") == "1"
# accepts both the MEMTIDE_* names and the plain .env names (LLM_API_KEY etc.)
LLM_URL = os.environ.get("MEMTIDE_LLM_BASE_URL") or os.environ.get("LLM_BASE_URL") or ""
LLM_KEY = os.environ.get("MEMTIDE_LLM_KEY") or os.environ.get("LLM_API_KEY") or ""
LLM_MODEL = os.environ.get("MEMTIDE_LLM_MODEL", "GLM-5.3-Flash")
DS_KEY = os.environ.get("MEMTIDE_DASHSCOPE_KEY") or os.environ.get("DASHSCOPE_API_KEY") or ""
RUN = uuid.uuid4().hex[:8]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@unittest.skipUnless(LIVE and LLM_KEY and DS_KEY,
                     "set MEMTIDE_LIVE=1 and provide real endpoint keys to run live tests")
class TestLiveEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from memtide import MemoryConfig, MemoryEngine
        from tests.testinfra import new_schema_dsn

        pg_dsn, _ = new_schema_dsn()
        cfg = MemoryConfig(
            storage_backend="postgres",
            pg_dsn=pg_dsn,
            vector_backend="qdrant",
            qdrant_url=os.environ.get("MEMTIDE_QDRANT_URL", "http://localhost:6333"),
            qdrant_collection=f"memtide_live_{RUN}",
            llm_backend="openai", llm_base_url=LLM_URL, llm_model=LLM_MODEL,
            llm_api_key=LLM_KEY,
            embedding_backend="dashscope", dashscope_api_key=DS_KEY,
        )
        cls.eng = MemoryEngine(cfg)
        cls.users = {k: f"live-{k}-{RUN}" for k in ("facts", "vectors", "cross", "slot", "consolidate", "audit", "image", "image-search")}

    @classmethod
    def tearDownClass(cls):
        cls.eng.close()

    def test_1_llm_extraction(self):
        r = self.eng.add("我叫李雷，住在杭州，喜欢喝美式咖啡", user_id=self.users["facts"])
        self.assertGreaterEqual(len(r.facts), 2, r.to_dict())
        self.assertGreaterEqual(len(r.added), 2)
        joined = " ".join(r.facts).lower()
        self.assertTrue("name" in joined or "名字" in joined, r.facts)

    def test_2_real_vector_search(self):
        self.eng.add("我叫李雷，住在杭州，喜欢喝美式咖啡", user_id=self.users["vectors"])
        hits = self.eng.search("用户住在哪里？", user_id=self.users["vectors"])
        self.assertTrue(hits)
        top = hits[0].memory.text.lower()
        self.assertTrue("hangzhou" in top or "杭州" in top, top)

    def test_3_cross_language_retrieval(self):
        """Real embeddings should retrieve English facts from Chinese queries."""
        self.eng.add("I work as a data engineer at Acme Corp", user_id=self.users["cross"])
        hits = self.eng.search("用户是做什么工作的", user_id=self.users["cross"])
        self.assertTrue(hits)
        self.assertIn("engineer", hits[0].memory.text.lower())

    def test_4_volatile_slot_update(self):
        self.eng.add("我是后端工程师", user_id=self.users["slot"])
        r = self.eng.add("我转行做产品经理了", user_id=self.users["slot"])
        self.assertEqual(len(r.updated), 1, r.to_dict())
        self.assertFalse(any("后端" in m.text or "backend" in m.text.lower()
                             for m in self.eng.get_all("live4")))

    def test_5_consolidation_with_llm(self):
        # infer=False seeds the cluster deterministically (LLM extraction
        # wording varies run to run); the summary itself still uses the real LLM
        for t in ["用户在用 Rust 写 CLI 工具", "用户在用 Rust 学 async 编程",
                  "用户在用 Rust 做练习项目"]:
            self.eng.add(t, user_id=self.users["consolidate"], infer=False)
        rep = self.eng.consolidate_background(user_id=self.users["consolidate"])
        self.assertEqual(rep["clusters"], 1, rep)
        self.assertGreaterEqual(rep["members_absorbed"], 3)
        self.assertIn("Rust", rep["summaries"][0]["text"])
        # originals superseded, bank shrank to summary (+ any off-topic leftovers)
        self.assertLessEqual(len(self.eng.get_all("live5")), 2)

    def test_6_audit_chain(self):
        r = self.eng.add("我叫王小明", user_id=self.users["audit"])
        mid = r.added[0]
        hist = self.eng.get_history(mid)
        self.assertTrue(any(e["event"] == "ADD" for e in hist))

    def test_7_multimodal_image(self):
        """Real image through the full pipeline with the REAL vision endpoint."""
        import base64
        import os
        import tempfile

        from .test_memtide import make_test_png

        png = make_test_png()
        data_url = "data:image/png;base64," + base64.b64encode(png).decode()
        media_dir = tempfile.mkdtemp(prefix="memtide-live-media-")
        self.eng.cfg.media_dir = media_dir
        r = self.eng.add(
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": "请记住这张图"},
            ]}],
            user_id=self.users["image"])
        self.assertTrue(r.attachments, "attachment must be recorded")
        att = r.attachments[0]
        self.assertEqual(att["kind"], "image")
        self.assertTrue(os.path.isfile(os.path.join(media_dir, att["source"])))
        print(f"\n  vision description: {att['description']!r}")

    def test_8_multimodal_retrieval_with_real_vision(self):
        """A text query must retrieve the image via its vision description."""
        import base64
        import os
        import tempfile

        from .test_memtide import make_test_png

        png = make_test_png()
        data_url = "data:image/png;base64," + base64.b64encode(png).decode()
        self.eng.cfg.media_dir = tempfile.mkdtemp(prefix="memtide-live-media-")
        r = self.eng.add(
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_url}}]}],
            user_id=self.users["image-search"])
        att = (r.attachments or [{}])[0]
        if not att.get("description"):
            self.skipTest("vision endpoint unavailable (graceful degradation path)")
        print(f"\n  vision description: {att['description']!r}")
        hits = self.eng.search("红色 色块", user_id=self.users["image-search"])
        self.assertTrue(hits)
        self.assertTrue(any(h.memory.attachments for h in hits))


if __name__ == "__main__":
    unittest.main(verbosity=2)

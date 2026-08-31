"""Quickstart: multi-turn memory with conflict resolution, retrieval, audit.

Reads endpoints/keys from the environment (source .env first). Expects a
reachable PostgreSQL via MEMTIDE_PG_DSN.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))

from memtide import MemoryEngine, MemoryConfig
from memtide.config import config_from_env

mem = MemoryEngine(config_from_env())
UID = "alice"

# ---- Day 1: introduction ----------------------------------------------------
print("=== 第一天：自我介绍 ===")
res = mem.add(
    [{"role": "user",
      "content": "嗨，我叫李雷，是一名后端工程师，住在杭州，喜欢喝美式咖啡，不喜欢加班，正在学习 Rust"}],
    user_id=UID,
)
for f in res.facts:
    print("  记住:", f)

# ---- Day 30: life changes ---------------------------------------------------
print("\n=== 第三十天：变化 ===")
res = mem.add("我搬到上海了，换到一家 AI 创业公司工作，现在喜欢加班了（ ProjX 太有意思）", user_id=UID)
print("  ops:", res.to_dict())

# ---- Retrieval with explainable scores ---------------------------------------
print("\n=== 检索：用户现在住哪？ ===")
for h in mem.search("用户现在住在哪个城市？", user_id=UID, limit=2):
    print(f"  [{h.score:.3f}] {h.memory.text}  {h.components}")

print("\n=== 检索：用户学什么语言？ ===")
for h in mem.search("用户在学什么编程语言", user_id=UID, limit=2):
    print(f"  [{h.score:.3f}] {h.memory.text}")

# ---- Inject into system prompt (Letta-style core block) ----------------------
print("\n=== 注入 system prompt 的记忆块 ===")
print(mem.render_context(user_id=UID, query="用户的近况"))

# ---- Temporal audit (Zep-style) ----------------------------------------------
print("\n=== 某条记忆的完整时间线 ===")
hit = mem.search("用户住在哪", user_id=UID, limit=1)[0]
for e in mem.get_history(memory_id=hit.memory.id):
    prev = (e["prev_value"] or "")[:20]
    new = (e["new_value"] or "")[:20]
    print(f"  {e['created_at']}  {e['event']:7s} {prev!r} -> {new!r}")
mem.close()

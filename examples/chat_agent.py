"""A minimal chat agent with long-term memory, showing the integration points
most agent frameworks need:

1. before each turn:  render_context() -> inject into system prompt
2. after each turn:   add()            -> extract & consolidate memories
3. relevance recall:  search()         -> query-scoped memory hits
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..'))

from memtide import MemoryEngine, MemoryConfig

from memtide.config import config_from_env

mem = MemoryEngine(config_from_env())
USER = "demo-user"


def respond(user_text: str) -> str:
    # 1) recall memories relevant to this specific question
    hits = mem.search(user_text, user_id=USER, limit=3)
    recall = "\n".join(f"- {h.memory.text}" for h in hits) or "（无相关记忆）"

    # 2) build the prompt a real agent would send to its LLM
    system_block = mem.render_context(user_id=USER)
    prompt = (
        f"{system_block}\n\n"
        f"## Recalled for this question\n{recall}\n\n"
        f"User: {user_text}\nAssistant:"
    )
    print(f"--- prompt sent to LLM ---\n{prompt}\n--------------------------")

    # In production, call your LLM here. For the demo we echo a template reply.
    reply = f"（演示回复，参考了 {len(hits)} 条相关记忆）"

    # 3) persist this turn's memory-worthy content
    mem.add(user_text, user_id=USER)
    return reply


if __name__ == "__main__":
    print("Assistant:", respond("我叫王小明，是个前端工程师，喜欢用 TypeScript"))
    print("Assistant:", respond("你还记得我是做什么的吗？"))
    print("Assistant:", respond("我最近转行做产品经理了"))
    mem.close()

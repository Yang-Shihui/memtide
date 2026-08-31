"""Seed rich demo data into a running Memtide stack (idempotent).

Designed to run INSIDE the memtide Docker container (has PG/Qdrant/keys):
    docker exec memtide-memtide-1 python /app/scripts/seed_demo.py

Uses the deployment's real LLM, embedding, PostgreSQL and Qdrant backends;
caption fields keep demo image descriptions deterministic, so the UI shows every feature:
multi-user scoping, slot conflicts with UPDATE audit chains, polarity
conflicts, episodic memories, a consolidation summary, invalidated memories.

Wipes the demo users first — safe to re-run, does not touch other users.
"""

import base64
import os
import struct
import sys
import zlib

sys.path.insert(0, "/app")

from memtide import MemoryConfig, MemoryEngine  # noqa: E402

DEMO_USERS = ["demo", "helen", "ken"]


def color_png(w, h, rgb):
    """A solid-color PNG, stdlib only — for demo image memories."""
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))

    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def main():
    # inherit the production deployment config (PG/Qdrant/real LLM/embedding)
    from memtide.config import config_from_env

    # this script PHYSICALLY deletes everything under the demo user names;
    # never do that silently on an interactive production deployment
    if "--yes" not in sys.argv and sys.stdin.isatty():
        reply = input(f"hard-delete ALL memories for users {DEMO_USERS}? type 'yes': ")
        if reply.strip().lower() != "yes":
            sys.exit("aborted")

    cfg = config_from_env()
    eng = MemoryEngine(cfg)
    print(f"backend: storage={cfg.storage_backend} vector={cfg.vector_backend} "
          f"embedder={cfg.embedding_backend}")

    # ---- clean slate for demo users -----------------------------------------
    for u in DEMO_USERS:
        for m in eng.store.all_rows(user_id=u, include_invalid=True):
            eng.vector_store.delete(m.id)
            eng.store.hard_delete(m.id)
    print("cleaned demo users")

    U = "demo"

    # ---- round 1: introduction (novel memories) ------------------------------
    r = eng.add("我叫林小满，是一名后端工程师，住在杭州，喜欢喝美式咖啡，不喜欢加班",
                user_id=U)
    print(f"round1: added={len(r.added)} rejected={len(r.rejected)}")

    # ---- round 2: episodic life ----------------------------------------------
    for t in ["周末去吃了川菜水煮鱼，特别辣",
              "和朋友去吃了川菜火锅，很开心"]:
        eng.add(t, user_id=U)
    # Rust 学习簇（infer=False 确定性原文直存，真实语义向量下彼此 cos≈0.6-0.75，
    # 恰好落在"整合区"：既不被门控当冗余拦下，又能被反思聚成一簇）
    for t in ["我正在用 Rust 写一个日志解析器",
              "我在学习 Rust 的 async 运行时",
              "我每天都会练习写 Rust 代码"]:
        eng.add(t, user_id=U, infer=False)

    # ---- round 3: life changes (volatile slot -> UPDATE audit chain) ----------
    r = eng.add("我搬到上海了", user_id=U)
    print(f"move: updated={len(r.updated)} (location slot conflict)")

    # ---- round 4: polarity conflict ------------------------------------------
    eng.add("我现在不讨厌加班了，新项目很有意思", user_id=U)

    # ---- round 5: reflection -> consolidated summary --------------------------
    rep = eng.consolidate_background(user_id=U)
    print(f"reflection: {rep['clusters']} cluster(s), {rep['members_absorbed']} absorbed")

    # ---- round 6: a soft-deleted memory (invalidated style in UI) --------------
    eng.add("记住：临时项目代号是 Project Kestrel", user_id=U, infer=False)
    for m in eng.get_all(U):
        if "Kestrel" in m.text:
            eng.delete(m.id)  # soft delete: shows strikethrough + audit trail
    print("soft-deleted one memory")

    # ---- multimodal: image memories with attachment previews -------------------
    # deterministic captions (no vision call) so the demo is stable; UI renders
    # the stored asset as a thumbnail via GET /media/{sha256}
    images = [
        (color_png(160, 120, (143, 203, 168)), "image",
         "一张开心果绿色的纯色方块图，配色参考马卡龙甜点"),
        (color_png(160, 120, (245, 168, 188)), "image",
         "一张草莓粉色的纯色方块图，柔和的粉彩配色"),
        (color_png(160, 120, (201, 182, 228)), "image",
         "一张薰衣草紫色的纯色方块图，浅紫粉彩色调"),
    ]
    for png, kind, caption in images:
        url = "data:image/png;base64," + base64.b64encode(png).decode()
        eng.add([{"role": "user", "content": [
            {"type": "text", "text": f"帮我记下这张图"},
            {"type": "image_url", "image_url": {"url": url}, "caption": caption},
        ]}], user_id=U)
    print(f"seeded {len(images)} image memories")

    # ---- historical import: data carries its own timestamp ---------------------
    eng.add("我那时候还在苏州出差，天天吃苏式面", user_id=U, infer=False,
            timestamp="2024-06-15T14:30:00+08:00")
    print("imported one 2024 memory (retention-decayed, hidden from default search)")

    # ---- side users: scope isolation demo --------------------------------------
    eng.add("我叫 Helen，是产品经理，喜欢骑行", user_id="helen")
    eng.add("我叫 Ken，在学烘焙，住在成都", user_id="ken")

    total = len(eng.get_all(U, limit=1000))
    allu = eng.store.stats()
    print(f"done: demo has {total} active memories; "
          f"stack total {allu['active_memories']} active / "
          f"{allu['invalidated_memories']} invalidated")
    eng.close()


if __name__ == "__main__":
    main()

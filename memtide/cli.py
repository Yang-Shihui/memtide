"""memtide CLI — manage the memory store from the shell (env-configured).

    python -m memtide add "我叫李雷，住在杭州"           # write memories
    python -m memtide search "用户住哪"                  # hybrid retrieval
    python -m memtide list                               # dump all memories
    python -m memtide context "用户喜欢什么咖啡"          # render core block
    python -m memtide history                            # audit log
    python -m memtide stats
    python -m memtide delete <memory_id>                 # soft-delete one memory
    python -m memtide serve --port 8300                  # start the REST API server
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import MemoryConfig
from .engine import MemoryEngine


def _engine(args) -> MemoryEngine:
    from .config import config_from_env

    return MemoryEngine(config_from_env())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="memtide", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("add", help="store memories from a conversation string")
    sp.add_argument("text")
    sp.add_argument("--user", default="default")

    sp = sub.add_parser("search", help="hybrid retrieval")
    sp.add_argument("query")
    sp.add_argument("--user", default="default")
    sp.add_argument("-k", type=int, default=5)

    sp = sub.add_parser("list", help="dump valid memories")
    sp.add_argument("--user", default="default")

    sp = sub.add_parser("context", help="render core memory block")
    sp.add_argument("query", nargs="?", default=None)
    sp.add_argument("--user", default="default")

    sub.add_parser("history", help="audit log")
    sub.add_parser("stats")

    sp = sub.add_parser("delete", help="soft-delete one memory")
    sp.add_argument("memory_id")

    sp = sub.add_parser("serve", help="start the REST API server")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8300)

    args = p.parse_args(argv)
    engine = _engine(args)

    try:
        if args.cmd == "serve":
            engine.close()
            from .server import serve_forever

            serve_forever(host=args.host, port=args.port)
            return 0
        if args.cmd == "add":
            res = engine.add(args.text, user_id=args.user)
            print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
        elif args.cmd == "search":
            hits = engine.search(args.query, user_id=args.user, limit=args.k)
            for h in hits:
                print(json.dumps(h.to_dict(), ensure_ascii=False))
        elif args.cmd == "list":
            for m in engine.get_all(args.user):
                print(json.dumps(m.to_dict(), ensure_ascii=False))
        elif args.cmd == "context":
            print(engine.render_context(user_id=args.user, query=args.query))
        elif args.cmd == "history":
            for h in engine.get_history(limit=50):
                print(json.dumps(h, ensure_ascii=False))
        elif args.cmd == "stats":
            print(json.dumps(engine.stats(), ensure_ascii=False, indent=2))
        elif args.cmd == "delete":
            ok = engine.delete(args.memory_id)
            print("deleted" if ok else "not found")
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

import os
import sys
import json
import argparse
from typing import List

from .manager import MemoryManager


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pinak-memory", description="Pinak Memory CLI")
    parser.add_argument("command", choices=["add", "search", "health"], help="Command to run")
    parser.add_argument("value", nargs="?", help="Content (for add) or query (for search)")
    parser.add_argument("--tags", nargs="*", default=[], help="Tags for add")
    parser.add_argument("--url", default=os.getenv("PINAK_MEMORY_URL", "http://localhost:8001"), help="Memory API base URL")
    parser.add_argument("--token", default=os.getenv("PINAK_TOKEN"), help="Bearer token")
    args = parser.parse_args(argv)

    mm = MemoryManager(service_base_url=args.url, token=args.token)

    if args.command == "health":
        ok = mm.health()
        print(json.dumps({"ok": ok}))
        return 0 if ok else 1
    if args.command == "add":
        if not args.value:
            print("content required", file=sys.stderr)
            return 2
        res = mm.add_memory(args.value, args.tags)
        if res is None:
            return 1
        print(json.dumps(res))
        return 0
    if args.command == "search":
        if not args.value:
            print("query required", file=sys.stderr)
            return 2
        res = mm.search_memory(args.value)
        print(json.dumps(res))
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


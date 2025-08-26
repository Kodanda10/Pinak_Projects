from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from .manager import MemoryManager
try:
    from ..bridge.context import ProjectContext
except Exception:
    ProjectContext = None  # type: ignore


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pinak-memory", description="Pinak Memory CLI")
    parser.add_argument("command", choices=["add", "search", "health"], help="Command to run")
    parser.add_argument("value", nargs="?", help="Content (for add) or query (for search)")
    parser.add_argument("--tags", nargs="*", default=[], help="Tags for add")
    parser.add_argument("--url", default=os.getenv("PINAK_MEMORY_URL"), help="Memory API base URL")
    parser.add_argument("--token", default=os.getenv("PINAK_TOKEN"), help="Bearer token")
    parser.add_argument("--project", default=os.getenv("PINAK_PROJECT_ID"), help="Project ID (from .pinak/pinak.json)")
    args = parser.parse_args(argv)

    # Auto-discover from Bridge context if not provided
    if ProjectContext and (not args.url or not args.token or not args.project):
        ctx = ProjectContext.find()
        if ctx:
            args.url = args.url or ctx.memory_url
            args.token = args.token or ctx.get_token()
            args.project = args.project or ctx.project_id

    mm = MemoryManager(service_base_url=args.url or "http://localhost:8001", token=args.token, project_id=args.project)

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


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
    parser.add_argument("command", 
                       choices=["add", "search", "health", "episodic", "procedural", "rag", "events", "session", "working", "search-all"], 
                       help="Command to run")
    parser.add_argument("value", nargs="?", help="Content (for add) or query (for search)")
    parser.add_argument("--tags", nargs="*", default=[], help="Tags for add")
    parser.add_argument("--url", default=os.getenv("PINAK_MEMORY_URL"), help="Memory API base URL")
    parser.add_argument("--token", default=os.getenv("PINAK_TOKEN"), help="Bearer token")
    parser.add_argument("--project", default=os.getenv("PINAK_PROJECT_ID"), help="Project ID (from .pinak/pinak.json)")
    parser.add_argument("--session-id", help="Session ID for session operations")
    parser.add_argument("--skill-id", help="Skill ID for procedural operations")
    parser.add_argument("--steps", nargs="*", help="Steps for procedural operations")
    parser.add_argument("--external-source", help="External source for RAG operations")
    parser.add_argument("--layers", default="episodic,procedural,rag", help="Layers for search-all")
    parser.add_argument("--limit", type=int, default=100, help="Limit for list operations")
    parser.add_argument("--ttl", type=int, help="TTL in seconds for session/working memory")
    args = parser.parse_args(argv)

    # Auto-discover from Bridge context if not provided
    if ProjectContext and (not args.url or not args.token or not args.project):
        ctx = ProjectContext.find()
        if ctx:
            args.url = args.url or ctx.memory_url
            args.token = args.token or ctx.get_token()
            args.project = args.project or ctx.project_id

    mm = MemoryManager(service_base_url=args.url or "http://localhost:8000", token=args.token, project_id=args.project)

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

    # ===== NEW 8-LAYER COMMANDS =====
    
    if args.command == "episodic":
        if not args.value:
            print("content required", file=sys.stderr)
            return 2
        salience = 0
        if args.tags and args.tags[0].isdigit():
            salience = int(args.tags[0])
        res = mm.add_episodic(args.value, salience)
        if res is None:
            return 1
        print(json.dumps(res))
        return 0
        
    if args.command == "procedural":
        if not args.skill_id or not args.steps:
            print("skill-id and steps required", file=sys.stderr)
            return 2
        res = mm.add_procedural(args.skill_id, args.steps)
        if res is None:
            return 1
        print(json.dumps(res))
        return 0
        
    if args.command == "rag":
        if not args.value:
            print("query required", file=sys.stderr)
            return 2
        res = mm.add_rag(args.value, args.external_source or "")
        if res is None:
            return 1
        print(json.dumps(res))
        return 0
        
    if args.command == "events":
        res = mm.list_events(limit=args.limit)
        print(json.dumps(res))
        return 0
        
    if args.command == "session":
        if not args.session_id or not args.value:
            print("session-id and content required", file=sys.stderr)
            return 2
        if "list" in (args.tags or []):
            res = mm.list_session(args.session_id, args.limit)
        else:
            res = mm.add_session(args.session_id, args.value, args.ttl)
        if res is None:
            return 1
        print(json.dumps(res))
        return 0
        
    if args.command == "working":
        if not args.value:
            print("content required", file=sys.stderr)
            return 2
        if "list" in (args.tags or []):
            res = mm.list_working(args.limit)
        else:
            res = mm.add_working(args.value, args.ttl)
        if res is None:
            return 1
        print(json.dumps(res))
        return 0
        
    if args.command == "search-all":
        if not args.value:
            print("query required", file=sys.stderr)
            return 2
        res = mm.search_all_layers(args.value, args.layers, args.limit)
        print(json.dumps(res))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from .manager import MemoryManager

try:
    from ..bridge.context import ProjectContext
except Exception:
    ProjectContext = None  # type: ignore


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="pinak-memory", description="Pinak Memory CLI")

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Health command
    health_parser = subparsers.add_parser("health", help="Check memory service health")
    health_parser.add_argument(
        "--url", default=os.getenv("PINAK_MEMORY_URL"), help="Memory API base URL"
    )
    health_parser.add_argument("--token", default=os.getenv("PINAK_TOKEN"), help="Bearer token")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search memory")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--url", default=os.getenv("PINAK_MEMORY_URL"), help="Memory API base URL"
    )
    search_parser.add_argument("--token", default=os.getenv("PINAK_TOKEN"), help="Bearer token")
    search_parser.add_argument(
        "--layers", default="episodic,procedural,rag", help="Layers for search"
    )
    search_parser.add_argument("--limit", type=int, default=100, help="Limit for results")

    # Add command
    add_parser = subparsers.add_parser("add", help="Add memory")
    add_parser.add_argument("content", help="Content to add")
    add_parser.add_argument(
        "--layer",
        required=True,
        choices=["episodic", "procedural", "rag", "events", "session", "working"],
        help="Memory layer",
    )
    add_parser.add_argument(
        "--url", default=os.getenv("PINAK_MEMORY_URL"), help="Memory API base URL"
    )
    add_parser.add_argument("--token", default=os.getenv("PINAK_TOKEN"), help="Bearer token")
    add_parser.add_argument(
        "--salience",
        type=int,
        default=5,
        help="Salience score for episodic memory (0-10)",
    )
    add_parser.add_argument("--skill-id", help="Skill ID for procedural operations")
    add_parser.add_argument("--steps", nargs="*", help="Steps for procedural operations")
    add_parser.add_argument("--external-source", help="External source for RAG operations")
    add_parser.add_argument("--session-id", help="Session ID for session operations")
    add_parser.add_argument("--ttl", type=int, help="TTL in seconds for session/working memory")

    # Layer-specific list commands
    episodic_parser = subparsers.add_parser("episodic", help="List episodic memories")
    episodic_parser.add_argument("--limit", type=int, default=100, help="Limit for results")
    episodic_parser.add_argument(
        "--url", default=os.getenv("PINAK_MEMORY_URL"), help="Memory API base URL"
    )
    episodic_parser.add_argument("--token", default=os.getenv("PINAK_TOKEN"), help="Bearer token")

    procedural_parser = subparsers.add_parser("procedural", help="List procedural memories")
    procedural_parser.add_argument("--limit", type=int, default=100, help="Limit for results")
    procedural_parser.add_argument(
        "--url", default=os.getenv("PINAK_MEMORY_URL"), help="Memory API base URL"
    )
    procedural_parser.add_argument("--token", default=os.getenv("PINAK_TOKEN"), help="Bearer token")

    rag_parser = subparsers.add_parser("rag", help="List RAG memories")
    rag_parser.add_argument("--limit", type=int, default=100, help="Limit for results")
    rag_parser.add_argument(
        "--url", default=os.getenv("PINAK_MEMORY_URL"), help="Memory API base URL"
    )
    rag_parser.add_argument("--token", default=os.getenv("PINAK_TOKEN"), help="Bearer token")

    events_parser = subparsers.add_parser("events", help="List events")
    events_parser.add_argument("--limit", type=int, default=100, help="Limit for results")
    events_parser.add_argument(
        "--url", default=os.getenv("PINAK_MEMORY_URL"), help="Memory API base URL"
    )
    events_parser.add_argument("--token", default=os.getenv("PINAK_TOKEN"), help="Bearer token")

    session_parser = subparsers.add_parser("session", help="List session memories")
    session_parser.add_argument("--session-id", help="Session ID")
    session_parser.add_argument("--limit", type=int, default=100, help="Limit for results")
    session_parser.add_argument(
        "--url", default=os.getenv("PINAK_MEMORY_URL"), help="Memory API base URL"
    )
    session_parser.add_argument("--token", default=os.getenv("PINAK_TOKEN"), help="Bearer token")

    working_parser = subparsers.add_parser("working", help="List working memories")
    working_parser.add_argument("--limit", type=int, default=100, help="Limit for results")
    working_parser.add_argument(
        "--url", default=os.getenv("PINAK_MEMORY_URL"), help="Memory API base URL"
    )
    working_parser.add_argument("--token", default=os.getenv("PINAK_TOKEN"), help="Bearer token")

    # Parse arguments
    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)

    # If no command provided, show help
    if not args.command:
        parser.print_help()
        return 0

    # Auto-discover from Bridge context if not provided
    if ProjectContext and (not getattr(args, "url", None) or not getattr(args, "token", None)):
        ctx = ProjectContext.find()
        if ctx:
            if hasattr(args, "url") and not args.url:
                args.url = ctx.memory_url
            if hasattr(args, "token") and not args.token:
                args.token = ctx.get_token()

    # Initialize MemoryManager
    url = getattr(args, "url", None) or os.getenv("PINAK_MEMORY_URL") or "http://localhost:8000"
    token = getattr(args, "token", None) or os.getenv("PINAK_TOKEN")
    mm = MemoryManager(service_base_url=url, token=token)

    # Handle commands
    if args.command == "health":
        ok = mm.health()
        print(json.dumps({"ok": ok}))
        return 0 if ok else 1

    elif args.command == "search":
        res = mm.search_all_layers(args.query, args.layers, args.limit)
        print(json.dumps(res))
        return 0

    elif args.command == "add":
        if args.layer == "episodic":
            res = mm.add_episodic(args.content, args.salience)
        elif args.layer == "procedural":
            if not args.skill_id or not args.steps:
                print("Procedural layer requires --skill-id and --steps", file=sys.stderr)
                return 2
            res = mm.add_procedural(args.skill_id, args.steps)
        elif args.layer == "rag":
            res = mm.add_rag(args.content, args.external_source or "")
        elif args.layer == "session":
            if not args.session_id:
                print("Session layer requires --session-id", file=sys.stderr)
                return 2
            res = mm.add_session(args.session_id, args.content, args.ttl)
        elif args.layer == "working":
            res = mm.add_working(args.content, args.ttl)
        else:
            print(f"Unsupported layer: {args.layer}", file=sys.stderr)
            return 2

        if res is None:
            return 1
        print(json.dumps(res))
        return 0

    elif args.command == "episodic":
        res = mm.list_episodic(args.limit)
        print(json.dumps(res))
        return 0

    elif args.command == "procedural":
        res = mm.list_procedural(args.limit)
        print(json.dumps(res))
        return 0

    elif args.command == "rag":
        res = mm.list_rag(args.limit)
        print(json.dumps(res))
        return 0

    elif args.command == "events":
        res = mm.list_events(limit=args.limit)
        print(json.dumps(res))
        return 0

    elif args.command == "session":
        session_id = args.session_id or ""
        res = mm.list_session(session_id, args.limit)
        print(json.dumps(res))
        return 0

    elif args.command == "working":
        res = mm.list_working(args.limit)
        print(json.dumps(res))
        return 0

    return 0

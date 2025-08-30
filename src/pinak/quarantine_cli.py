"""
CLI interface for the File Quarantine System.

Provides command-line tools for managing quarantined files, restoring files,
and monitoring the quarantine system.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List

from .file_quarantine import (
    QuarantineAction,
    QuarantinePriority,
    get_quarantine_manager,
    safe_delete,
    safe_move,
)


def cmd_quarantine(args: argparse.Namespace) -> int:
    """Handle file quarantine command."""
    if not args.file.exists():
        print(f"Error: File does not exist: {args.file}", file=sys.stderr)
        return 1

    manager = get_quarantine_manager()

    record_id = manager.quarantine_file(
        file_path=args.file,
        action=QuarantineAction(args.action),
        reason=args.reason,
        user=args.user,
        priority=QuarantinePriority(args.priority),
    )

    if record_id:
        print(f"File quarantined successfully. Record ID: {record_id}")
        return 0
    else:
        print("Failed to quarantine file", file=sys.stderr)
        return 1


def cmd_restore(args: argparse.Namespace) -> int:
    """Handle file restore command."""
    manager = get_quarantine_manager()

    success = manager.restore_file(args.record_id, args.target)

    if success:
        print(f"File restored successfully from record: {args.record_id}")
        return 0
    else:
        print(f"Failed to restore file from record: {args.record_id}", file=sys.stderr)
        return 1


def cmd_list(args: argparse.Namespace) -> int:
    """Handle list quarantined files command."""
    manager = get_quarantine_manager()

    # Parse filters
    action_filter = QuarantineAction(args.action) if args.action else None
    priority_filter = QuarantinePriority(args.priority) if args.priority else None

    records = manager.list_quarantined_files(
        action_filter=action_filter,
        priority_filter=priority_filter,
        user_filter=args.user,
        limit=args.limit,
    )

    if not records:
        print("No quarantined files found.")
        return 0

    # Display results
    print(f"Found {len(records)} quarantined files:")
    print("-" * 80)

    for record in records:
        print(f"ID: {record['record_id']}")
        print(f"Original: {record['original_path']}")
        print(f"Action: {record['action']}")
        print(f"Priority: {record['priority']}")
        print(f"User: {record['user']}")
        print(f"Timestamp: {record['timestamp']}")
        print(f"Size: {record['file_size']} bytes")
        print(f"Can Restore: {record['can_restore']}")
        if record["reason"]:
            print(f"Reason: {record['reason']}")
        print("-" * 80)

    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    """Handle quarantine statistics command."""
    manager = get_quarantine_manager()

    stats = manager.get_quarantine_stats()

    print("File Quarantine Statistics")
    print("=" * 40)
    print(f"Total Files: {stats['total_files']}")
    print(f"Total Size: {stats['total_size_gb']:.2f} GB")
    print(f"Auto Cleanup: {stats['auto_cleanup_days']} days")
    print(f"Max Size: {stats['max_size_gb']} GB")
    print()

    print("Priority Distribution:")
    for priority, count in stats["priority_distribution"].items():
        print(f"  {priority}: {count}")
    print()

    print("Action Distribution:")
    for action, count in stats["action_distribution"].items():
        print(f"  {action}: {count}")
    print()

    print("Age Distribution:")
    for age_range, count in stats["age_distribution"].items():
        print(f"  {age_range}: {count}")

    return 0


def cmd_cleanup(args: argparse.Namespace) -> int:
    """Handle cleanup command."""
    manager = get_quarantine_manager()

    stats = manager.cleanup_old_files(
        max_age_days=args.days,
        priority_filter=QuarantinePriority(args.priority) if args.priority else None,
        dry_run=args.dry_run,
    )

    action = "Would clean up" if args.dry_run else "Cleaned up"
    print(f"{action}: {stats['cleaned_up']} files")
    print(f"Total size freed: {stats['total_size_freed']} bytes")
    print(f"Errors: {stats['errors']}")

    if args.dry_run:
        print("Use --no-dry-run to actually perform cleanup")

    return 0


def cmd_safe_delete(args: argparse.Namespace) -> int:
    """Handle safe delete command."""
    success = safe_delete(args.file, args.reason, args.user or "cli_user")

    if success:
        print(f"File safely quarantined: {args.file}")
        return 0
    else:
        print(f"Failed to quarantine file: {args.file}", file=sys.stderr)
        return 1


def cmd_safe_move(args: argparse.Namespace) -> int:
    """Handle safe move command."""
    success = safe_move(args.src, args.dst, args.reason, args.user or "cli_user")

    if success:
        print(f"File safely moved: {args.src} -> {args.dst}")
        return 0
    else:
        print(f"Failed to move file: {args.src} -> {args.dst}", file=sys.stderr)
        return 1


def main(argv: List[str] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pinak-quarantine", description="File Quarantine System CLI for Pinak"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Quarantine command
    quarantine_parser = subparsers.add_parser("quarantine", help="Quarantine a file")
    quarantine_parser.add_argument("file", type=Path, help="File to quarantine")
    quarantine_parser.add_argument(
        "--action",
        choices=[a.value for a in QuarantineAction],
        default=QuarantineAction.DELETE_REQUESTED.value,
        help="Type of quarantine action",
    )
    quarantine_parser.add_argument("--reason", required=True, help="Reason for quarantine")
    quarantine_parser.add_argument("--user", default="cli_user", help="User performing the action")
    quarantine_parser.add_argument(
        "--priority",
        choices=[p.value for p in QuarantinePriority],
        default=QuarantinePriority.MEDIUM.value,
        help="Priority level",
    )
    quarantine_parser.set_defaults(func=cmd_quarantine)

    # Restore command
    restore_parser = subparsers.add_parser("restore", help="Restore a quarantined file")
    restore_parser.add_argument("record_id", help="Quarantine record ID")
    restore_parser.add_argument(
        "--target",
        type=Path,
        help="Target path for restoration (defaults to original path)",
    )
    restore_parser.set_defaults(func=cmd_restore)

    # List command
    list_parser = subparsers.add_parser("list", help="List quarantined files")
    list_parser.add_argument(
        "--action",
        choices=[a.value for a in QuarantineAction],
        help="Filter by action type",
    )
    list_parser.add_argument(
        "--priority",
        choices=[p.value for p in QuarantinePriority],
        help="Filter by priority",
    )
    list_parser.add_argument("--user", help="Filter by user")
    list_parser.add_argument("--limit", type=int, default=50, help="Maximum number of results")
    list_parser.set_defaults(func=cmd_list)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show quarantine statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old quarantined files")
    cleanup_parser.add_argument("--days", type=int, default=90, help="Maximum age in days")
    cleanup_parser.add_argument(
        "--priority",
        choices=[p.value for p in QuarantinePriority],
        help="Only cleanup specific priority",
    )
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without actually doing it",
    )
    cleanup_parser.set_defaults(func=cmd_cleanup)

    # Safe delete command
    safe_delete_parser = subparsers.add_parser(
        "safe-delete", help="Safely quarantine a file (like delete but recoverable)"
    )
    safe_delete_parser.add_argument("file", type=Path, help="File to quarantine")
    safe_delete_parser.add_argument("--reason", required=True, help="Reason for quarantine")
    safe_delete_parser.add_argument("--user", help="User performing the action")
    safe_delete_parser.add_argument(
        "--no-patch", action="store_true", help="Do not patch system file operations"
    )
    safe_delete_parser.set_defaults(func=cmd_safe_delete)

    # Safe move command
    safe_move_parser = subparsers.add_parser(
        "safe-move", help="Safely move a file with quarantine backup"
    )
    safe_move_parser.add_argument("src", type=Path, help="Source file")
    safe_move_parser.add_argument("dst", type=Path, help="Destination path")
    safe_move_parser.add_argument("--reason", required=True, help="Reason for move")
    safe_move_parser.add_argument("--user", help="User performing the action")
    safe_move_parser.set_defaults(func=cmd_safe_move)

    # Parse and execute
    args = parser.parse_args(argv)

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

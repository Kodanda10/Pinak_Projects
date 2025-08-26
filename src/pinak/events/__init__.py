"""
pinak.events
Lightweight event-layer helpers and models.

Provides:
- EventMemory model (from 7-layer schema)
- log_event(): append event to a JSONL hash-chained ledger
"""
from __future__ import annotations

from typing import Dict, Any, Optional
import json
from pathlib import Path

from pinak.memory.schemas import EventMemory
from pinak.ledger.hash_chain import append_entry


def _last_hash(path: Path) -> Optional[str]:
    try:
        if not path.exists():
            return None
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            block = 2048
            data = b""
            while size > 0 and b"\n" not in data:
                step = min(block, size)
                size -= step
                f.seek(size)
                data = f.read(block) + data
            last = data.splitlines()[-1] if data else b""
            if last:
                return json.loads(last.decode("utf-8")).get("hash")
    except Exception:
        return None
    return None


def log_event(entry: Dict[str, Any], ledger_path: str = "events.jsonl") -> str:
    """Append an event entry to a hash-chained JSONL ledger and return its hash."""
    p = Path(ledger_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    prev = _last_hash(p)
    return append_entry(str(p), entry, prev_hash=prev)

__all__ = [
    "EventMemory",
    "log_event",
]


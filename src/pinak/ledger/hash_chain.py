import json
import hashlib
from typing import Optional, Dict, Any


def append_entry(path: str, entry: Dict[str, Any], prev_hash: Optional[str] = None) -> str:
    """Append an entry to a JSONL ledger with a hash chain.

    Returns the new entry hash. The file is append-only; callers must not rewrite.
    """
    rec = {"entry": entry, "prev_hash": prev_hash}
    blob = json.dumps(rec, separators=(",", ":"), sort_keys=True)
    h = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    rec["hash"] = h
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    return h


def verify_chain(path: str) -> bool:
    prev: Optional[str] = None
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                expected_prev = prev
                if obj.get("prev_hash") != expected_prev:
                    return False
                # recompute
                tmp = {"entry": obj.get("entry"), "prev_hash": obj.get("prev_hash")}
                blob = json.dumps(tmp, separators=(",", ":"), sort_keys=True)
                h = hashlib.sha256(blob.encode("utf-8")).hexdigest()
                if obj.get("hash") != h:
                    return False
                prev = h
        return True
    except FileNotFoundError:
        return True


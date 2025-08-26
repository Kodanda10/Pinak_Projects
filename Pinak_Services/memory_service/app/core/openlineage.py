import json
from pathlib import Path
from typing import Dict, Any


def emit_openlineage(event: Dict[str, Any], path: str = "/code/data/openlineage.jsonl") -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception:
        # Best-effort; avoid breaking writes due to telemetry
        pass


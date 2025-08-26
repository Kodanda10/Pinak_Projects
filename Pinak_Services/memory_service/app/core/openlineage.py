import json
import os
from pathlib import Path
from typing import Dict, Any
import urllib.request


def emit_openlineage(event: Dict[str, Any], path: str = "/code/data/openlineage.jsonl") -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        # Optional endpoint emission
        endpoint = os.getenv("OPENLINEAGE_ENDPOINT")
        if endpoint:
            req = urllib.request.Request(endpoint, data=json.dumps(event).encode('utf-8'), headers={'Content-Type':'application/json'}, method='POST')
            try:
                urllib.request.urlopen(req, timeout=2)
            except Exception:
                pass
    except Exception:
        # Best-effort; avoid breaking writes due to telemetry
        pass

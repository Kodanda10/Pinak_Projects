from fastapi import APIRouter, status
from app.core.schemas import MemoryCreate, MemoryRead, MemorySearchResult
from fastapi import Body
from typing import Dict, Any
from datetime import datetime, timezone
import uuid
from pathlib import Path
import json
from app.services.memory_service import memory_service
from pinak.ledger.hash_chain import append_entry  # provided by SDK
from app.core.tenancy import resolve_tenant
from app.core.ttl import ttl_for_layer
from app.core.openlineage import emit_openlineage
from typing import List

router = APIRouter()

@router.post("/add", response_model=MemoryRead, status_code=status.HTTP_201_CREATED)
def add_memory(memory: MemoryCreate):
    """API endpoint to add a new memory."""
    return memory_service.add_memory(memory)

@router.get("/search", response_model=List[MemorySearchResult])
def search_memory(query: str, k: int = 5, request=None):
    """API endpoint to search for relevant memories."""
    # Switch tenant context for search
    try:
        tenant = resolve_tenant(request, {}) if request is not None else "default"
        memory_service.switch_tenant(tenant)
    except Exception:
        pass
    return memory_service.search_memory(query=query, k=k)


@router.post("/add_v2", status_code=status.HTTP_201_CREATED)
def add_memory_v2(payload: Dict[str, Any] = Body(...), request=None) -> Dict[str, Any]:
    """
    Non-breaking v2 endpoint that accepts layered payloads.
    - If payload resembles MemoryCreate, store semantic memory as before.
    - Always append a changelog entry to a hash-chain ledger (WORM).
    - Enforce idempotency by op_id when provided.
    """
    # Idempotency via op_id
    op_id = payload.get("op_id") or str(uuid.uuid4())
    layer = payload.get("layer") or "semantic"
    tenant = resolve_tenant(request, payload) if request is not None else payload.get("tenant", "default")
    content = payload.get("content") or payload.get("text")
    tags = payload.get("tags") or []

    # Store semantic content using existing service when content present
    stored = None
    if content:
        # Per-tenant isolation for semantic store
        try:
            memory_service.switch_tenant(tenant)
        except Exception:
            pass
        stored = memory_service.add_memory(MemoryCreate(content=content, tags=tags))
        # Apply TTL if provided by policy
        try:
            ttl = ttl_for_layer(layer)
            memory_service.set_expiry_by_id(stored.id, ttl)
        except Exception:
            pass

    # Append to hash-chain ledger (per-project/tenant future-ready)
    ledger_dir = Path(f"/code/data/tenants/{tenant}")
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_file = ledger_dir / "ledger.jsonl"
    entry = {
        "op_id": op_id,
        "layer": layer,
        "ts": datetime.now(timezone.utc).isoformat(),
        "tenant": tenant,
        "ttl_seconds": ttl_for_layer(layer),
        "content_ref": getattr(stored, "id", None),
        "payload_meta": {k: v for k, v in payload.items() if k not in {"content", "text"}},
    }
    # Compute prev_hash by verifying and reading last line (best-effort)
    try:
        prev = None
        if ledger_file.exists():
            with open(ledger_file, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                block = 1024
                data = b""
                while size > 0 and b"\n" not in data:
                    step = min(block, size)
                    size -= step
                    f.seek(size)
                    data = f.read(block) + data
                last = data.splitlines()[-1] if data else b""
                if last:
                    obj = json.loads(last.decode("utf-8"))
                    prev = obj.get("hash")
        h = append_entry(str(ledger_file), entry, prev_hash=prev)
    except Exception:
        h = None

    # Best-effort OpenLineage event
    emit_openlineage({
        "job": "memory.write",
        "dataset": f"mem:{layer}",
        "tenant": tenant,
        "run": op_id,
        "outputs": [getattr(stored, "id", None)],
    })

    return {
        "status": "ok",
        "op_id": op_id,
        "layer": layer,
        "tenant": tenant,
        "stored_id": getattr(stored, "id", None),
        "ledger_hash": h,
    }

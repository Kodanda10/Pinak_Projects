from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, status, HTTPException

from app.core.schemas import (
    MemoryCreate, MemoryRead, MemorySearchResult,
    EpisodicCreate, ProceduralCreate, RAGCreate, EventCreate, SessionCreate,
    ContextResponse, WorkingCreate
)
from app.core.security import AuthContext, require_auth_context
from app.services.memory_service import MemoryService

@lru_cache
def _service_factory() -> MemoryService:
    return MemoryService()

def get_memory_service() -> MemoryService:
    """Dependency that provides a cached MemoryService instance."""
    return _service_factory()

router = APIRouter()

# --- Semantic Memory ---

@router.post("/add", response_model=MemoryRead, status_code=status.HTTP_201_CREATED)
def add_memory(
    memory: MemoryCreate,
    ctx: AuthContext = Depends(require_auth_context),
    service: MemoryService = Depends(get_memory_service),
):
    """Add a semantic memory (Vector + DB)."""
    return service.add_memory(memory, ctx.tenant_id, ctx.project_id)

@router.get("/search", response_model=List[MemorySearchResult])
def search_memory(
    query: str,
    k: int = 5,
    ctx: AuthContext = Depends(require_auth_context),
    service: MemoryService = Depends(get_memory_service),
):
    """Legacy Semantic Search."""
    return service.search_memory(query, ctx.tenant_id, ctx.project_id, k)

# --- Unified Retrieval ---

@router.get("/retrieve_context", response_model=ContextResponse)
def retrieve_context(
    query: str,
    ctx: AuthContext = Depends(require_auth_context),
    service: MemoryService = Depends(get_memory_service),
):
    """
    Unified Endpoint for Agents.
    Performs Hybrid Search across all layers and returns categorized context.
    """
    return service.retrieve_context(query, ctx.tenant_id, ctx.project_id)

# --- Episodic Memory ---

@router.post("/episodic/add", status_code=status.HTTP_201_CREATED)
def add_episodic(
    item: EpisodicCreate,
    ctx: AuthContext = Depends(require_auth_context),
    service: MemoryService = Depends(get_memory_service),
):
    return service.add_episodic(
        item.content, ctx.tenant_id, ctx.project_id,
        item.salience, item.goal, item.plan, item.outcome, item.tool_logs
    )

# --- Procedural Memory ---

@router.post("/procedural/add", status_code=status.HTTP_201_CREATED)
def add_procedural(
    item: ProceduralCreate,
    ctx: AuthContext = Depends(require_auth_context),
    service: MemoryService = Depends(get_memory_service),
):
    return service.add_procedural(
        item.skill_name, item.steps, ctx.tenant_id, ctx.project_id,
        item.trigger, item.code_snippet
    )

# --- RAG ---

@router.post("/rag/add", status_code=status.HTTP_201_CREATED)
def add_rag(
    item: RAGCreate,
    ctx: AuthContext = Depends(require_auth_context),
    service: MemoryService = Depends(get_memory_service),
):
    return service.add_rag(
        item.query, item.external_source, item.content, ctx.tenant_id, ctx.project_id
    )

# --- Logs & Events ---

@router.post("/event", status_code=status.HTTP_201_CREATED)
def add_event(
    item: EventCreate,
    ctx: AuthContext = Depends(require_auth_context),
    service: MemoryService = Depends(get_memory_service),
):
    return service.add_event(
        {"event_type": item.event_type, **item.payload},
        ctx.tenant_id, ctx.project_id
    )

@router.get("/events")
def list_events(
    limit: int = 100,
    ctx: AuthContext = Depends(require_auth_context),
    service: MemoryService = Depends(get_memory_service),
):
    return service.list_events(ctx.tenant_id, ctx.project_id, limit)

@router.post("/session/add", status_code=status.HTTP_201_CREATED)
def add_session(
    item: SessionCreate,
    ctx: AuthContext = Depends(require_auth_context),
    service: MemoryService = Depends(get_memory_service),
):
    return service.session_add(
        item.session_id, item.content, item.role, ctx.tenant_id, ctx.project_id
    )

@router.get("/session/list")
def list_session(
    session_id: str,
    limit: int = 100,
    ctx: AuthContext = Depends(require_auth_context),
    service: MemoryService = Depends(get_memory_service),
):
    return service.session_list(session_id, ctx.tenant_id, ctx.project_id, limit)

@router.post("/working/add", status_code=status.HTTP_201_CREATED)
def add_working(
    item: WorkingCreate,
    ctx: AuthContext = Depends(require_auth_context),
    service: MemoryService = Depends(get_memory_service),
):
    return service.working_add(item.content, ctx.tenant_id, ctx.project_id)

@router.get("/working/list")
def list_working(
    limit: int = 100,
    ctx: AuthContext = Depends(require_auth_context),
    service: MemoryService = Depends(get_memory_service),
):
    return service.working_list(ctx.tenant_id, ctx.project_id, limit)

@router.put("/{layer}/{memory_id}", status_code=status.HTTP_200_OK)
def update_memory(
    layer: str,
    memory_id: str,
    updates: Dict[str, Any] = Body(...),
    ctx: AuthContext = Depends(require_auth_context),
    service: MemoryService = Depends(get_memory_service),
):
    success = service.update_memory(layer, memory_id, updates, ctx.tenant_id, ctx.project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "updated", "id": memory_id}

@router.delete("/{layer}/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_memory(
    layer: str,
    memory_id: str,
    ctx: AuthContext = Depends(require_auth_context),
    service: MemoryService = Depends(get_memory_service),
):
    success = service.delete_memory(layer, memory_id, ctx.tenant_id, ctx.project_id)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")

from fastapi import APIRouter, status, Header, HTTPException, Request, Body
import os
from jose import jwt, JWTError
from app.core.schemas import MemoryCreate, MemoryRead, MemorySearchResult
from app.services.memory_service import (
    memory_service,
    add_episodic as svc_add_episodic,
    list_episodic as svc_list_episodic,
    add_procedural as svc_add_procedural,
    list_procedural as svc_list_procedural,
    add_rag as svc_add_rag,
    list_rag as svc_list_rag,
    search_v2 as svc_search_v2,
)
def resolve_tenant(request, payload):
    try:
        # Best-effort tenant from header or payload
        if request is not None:
            v = request.headers.get('X-Pinak-Tenant')
            if v:
                return v
        return (payload or {}).get('tenant', 'default')
    except Exception:
        return 'default'
from typing import List, Dict, Any, Optional

router = APIRouter()

@router.post("/add", response_model=MemoryRead, status_code=status.HTTP_201_CREATED)
def add_memory(memory: MemoryCreate):
    """API endpoint to add a new memory."""
    return memory_service.add_memory(memory)

@router.get("/search", response_model=List[MemorySearchResult])
def search_memory(query: str, k: int = 5):
    """API endpoint to search for relevant memories."""
    return memory_service.search_memory(query=query, k=k)

@router.post("/episodic/add", status_code=status.HTTP_201_CREATED)
def add_episodic(payload: Dict[str, Any] = Body(...), request: Request = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> Dict[str, Any]:
    # Optional: if Authorization present, enforce pid == header
    try:
        auth = request.headers.get('Authorization') if request else None
        if auth and auth.lower().startswith('bearer '):
            token = auth.split(' ',1)[1]
            claims = jwt.decode(token, os.getenv('SECRET_KEY','change-me-in-prod'), algorithms=["HS256"])
            if project_id and claims.get('pid') and claims['pid'] != project_id:
                raise HTTPException(status_code=403, detail="Project header/token mismatch")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    tenant = resolve_tenant(request, payload) if request is not None else payload.get("tenant", "default")
    rec = svc_add_episodic(memory_service, tenant, project_id, payload.get('content') or '', int(payload.get('salience') or 0))
    return rec

@router.get("/episodic/list", status_code=status.HTTP_200_OK)
def list_episodic(request: Request = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> list[Dict[str, Any]]:
    tenant = resolve_tenant(request, {}) if request is not None else "default"
    return svc_list_episodic(memory_service, tenant, project_id)

@router.post("/procedural/add", status_code=status.HTTP_201_CREATED)
def add_procedural(payload: Dict[str, Any] = Body(...), request: Request = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> Dict[str, Any]:
    try:
        auth = request.headers.get('Authorization') if request else None
        if auth and auth.lower().startswith('bearer '):
            token = auth.split(' ',1)[1]
            claims = jwt.decode(token, os.getenv('SECRET_KEY','change-me-in-prod'), algorithms=["HS256"])
            if project_id and claims.get('pid') and claims['pid'] != project_id:
                raise HTTPException(status_code=403, detail="Project header/token mismatch")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    tenant = resolve_tenant(request, payload) if request is not None else payload.get("tenant", "default")
    rec = svc_add_procedural(memory_service, tenant, project_id, payload.get('skill_id') or 'skill', payload.get('steps') or [])
    return rec

@router.get("/procedural/list", status_code=status.HTTP_200_OK)
def list_procedural(request: Request = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> list[Dict[str, Any]]:
    tenant = resolve_tenant(request, {}) if request is not None else "default"
    return svc_list_procedural(memory_service, tenant, project_id)

@router.post("/rag/add", status_code=status.HTTP_201_CREATED)
def add_rag(payload: Dict[str, Any] = Body(...), request: Request = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> Dict[str, Any]:
    try:
        auth = request.headers.get('Authorization') if request else None
        if auth and auth.lower().startswith('bearer '):
            token = auth.split(' ',1)[1]
            claims = jwt.decode(token, os.getenv('SECRET_KEY','change-me-in-prod'), algorithms=["HS256"])
            if project_id and claims.get('pid') and claims['pid'] != project_id:
                raise HTTPException(status_code=403, detail="Project header/token mismatch")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    tenant = resolve_tenant(request, payload) if request is not None else payload.get("tenant", "default")
    rec = svc_add_rag(memory_service, tenant, project_id, payload.get('query') or '', payload.get('external_source'))
    return rec

@router.get("/rag/list", status_code=status.HTTP_200_OK)
def list_rag(request: Request = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> list[Dict[str, Any]]:
    tenant = resolve_tenant(request, {}) if request is not None else "default"
    return svc_list_rag(memory_service, tenant, project_id)

@router.get("/search_v2", status_code=status.HTTP_200_OK)
def search_v2(query: str, layers: str = 'semantic', request: Request = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> Dict[str, Any]:
    tenant = resolve_tenant(request, {}) if request is not None else "default"
    layer_list = [s.strip() for s in layers.split(',') if s.strip()]
    return svc_search_v2(memory_service, tenant, project_id, query, layer_list)

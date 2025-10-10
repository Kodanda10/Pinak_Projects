from functools import lru_cache
from typing import Any, Dict, List, Optional
import datetime

from fastapi import APIRouter, Body, Depends, status

from app.core.schemas import MemoryCreate, MemoryRead, MemorySearchResult
from app.core.security import AuthContext, require_auth_context
from app.services.memory_service import (
    MemoryService,
    add_episodic as svc_add_episodic,
    add_procedural as svc_add_procedural,
    add_rag as svc_add_rag,
    list_episodic as svc_list_episodic,
    list_procedural as svc_list_procedural,
    list_rag as svc_list_rag,
    search_v2 as svc_search_v2,
)


@lru_cache
def _service_factory() -> MemoryService:
    return MemoryService()


def get_memory_service() -> MemoryService:
    """Dependency that provides a cached MemoryService instance."""

    return _service_factory()


router = APIRouter()


@router.post("/add", response_model=MemoryRead, status_code=status.HTTP_201_CREATED)
def add_memory(
    memory: MemoryCreate,
    ctx: AuthContext = Depends(require_auth_context),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """API endpoint to add a new memory."""

    return memory_service.add_memory(memory, ctx.tenant_id, ctx.project_id)


@router.get("/search", response_model=List[MemorySearchResult])
def search_memory(
    query: str,
    k: int = 5,
    ctx: AuthContext = Depends(require_auth_context),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """API endpoint to search for relevant memories."""

    return memory_service.search_memory(query=query, tenant=ctx.tenant_id, project_id=ctx.project_id, k=k)

@router.post("/episodic/add", status_code=status.HTTP_201_CREATED)
def add_episodic(
    payload: Dict[str, Any] = Body(...),
    ctx: AuthContext = Depends(require_auth_context),
    memory_service: MemoryService = Depends(get_memory_service),
) -> Dict[str, Any]:
    return svc_add_episodic(
        memory_service,
        ctx.tenant_id,
        ctx.project_id,
        payload.get('content') or '',
        int(payload.get('salience') or 0),
    )

@router.get("/episodic/list", status_code=status.HTTP_200_OK)
def list_episodic(
    ctx: AuthContext = Depends(require_auth_context),
    memory_service: MemoryService = Depends(get_memory_service),
) -> List[Dict[str, Any]]:
    return svc_list_episodic(memory_service, ctx.tenant_id, ctx.project_id)

@router.post("/procedural/add", status_code=status.HTTP_201_CREATED)
def add_procedural(
    payload: Dict[str, Any] = Body(...),
    ctx: AuthContext = Depends(require_auth_context),
    memory_service: MemoryService = Depends(get_memory_service),
) -> Dict[str, Any]:
    return svc_add_procedural(
        memory_service,
        ctx.tenant_id,
        ctx.project_id,
        payload.get('skill_id') or 'skill',
        payload.get('steps') or [],
    )


@router.get("/procedural/list", status_code=status.HTTP_200_OK)
def list_procedural(
    ctx: AuthContext = Depends(require_auth_context),
    memory_service: MemoryService = Depends(get_memory_service),
) -> List[Dict[str, Any]]:
    return svc_list_procedural(memory_service, ctx.tenant_id, ctx.project_id)

@router.post("/rag/add", status_code=status.HTTP_201_CREATED)
def add_rag(
    payload: Dict[str, Any] = Body(...),
    ctx: AuthContext = Depends(require_auth_context),
    memory_service: MemoryService = Depends(get_memory_service),
) -> Dict[str, Any]:
    return svc_add_rag(
        memory_service,
        ctx.tenant_id,
        ctx.project_id,
        payload.get('query') or '',
        payload.get('external_source') or '',
    )


@router.get("/rag/list", status_code=status.HTTP_200_OK)
def list_rag(
    ctx: AuthContext = Depends(require_auth_context),
    memory_service: MemoryService = Depends(get_memory_service),
) -> List[Dict[str, Any]]:
    return svc_list_rag(memory_service, ctx.tenant_id, ctx.project_id)

@router.get("/search_v2", status_code=status.HTTP_200_OK)
def search_v2(
    query: str,
    layers: str = 'episodic',
    limit: int = 20,
    offset: int = 0,
    ctx: AuthContext = Depends(require_auth_context),
    memory_service: MemoryService = Depends(get_memory_service),
) -> Dict[str, Any]:
    layer_list = [s.strip() for s in layers.split(',') if s.strip()]
    res = svc_search_v2(memory_service, ctx.tenant_id, ctx.project_id, query, layer_list)
    for k,v in list(res.items()):
        if isinstance(v, list):
            res[k] = v[offset:offset+limit]
    return res

@router.post("/event", status_code=status.HTTP_201_CREATED, response_model=None)
def add_event(
    payload: Dict[str, Any] = Body(...),
    ctx: AuthContext = Depends(require_auth_context),
    memory_service: MemoryService = Depends(get_memory_service),
) -> Dict[str, Any]:
    base = memory_service._store_dir(ctx.tenant_id, ctx.project_id)
    event_payload = {
        "ts": datetime.datetime.utcnow().isoformat(),
        **payload,
        "tenant": ctx.tenant_id,
        "project_id": ctx.project_id,
    }
    ep = memory_service._dated_file(base, 'events', 'events')
    memory_service._append_audit_jsonl(ep, event_payload)
    return {"status": "ok"}

@router.get("/events", status_code=status.HTTP_200_OK, response_model=None)
def list_events(
    q: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    ctx: AuthContext = Depends(require_auth_context),
    memory_service: MemoryService = Depends(get_memory_service),
) -> List[Dict[str, Any]]:
    import json, os, datetime
    base = memory_service._store_dir(ctx.tenant_id, ctx.project_id)
    out=[]
    def parse_ts(ts: str):
        try:
            return datetime.datetime.fromisoformat(ts.replace('Z','+00:00'))
        except Exception:
            return None
    t_since = parse_ts(since) if since else None
    t_until = parse_ts(until) if until else None
    import glob
    folder = os.path.join(base, 'events')
    for fp in sorted(glob.glob(os.path.join(folder, 'events_*.jsonl'))):
        if os.path.exists(fp):
            with open(fp,'r',encoding='utf-8') as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        if q and q not in json.dumps(obj):
                            continue
                        ts = parse_ts(obj.get('ts',''))
                        if t_since and ts and ts < t_since:
                            continue
                        if t_until and ts and ts > t_until:
                            continue
                        out.append(obj)
                    except Exception:
                        pass
    return out[offset:offset+limit]

@router.post("/session/add", status_code=status.HTTP_201_CREATED, response_model=None)
def session_add(
    payload: Dict[str, Any] = Body(...),
    ctx: AuthContext = Depends(require_auth_context),
    memory_service: MemoryService = Depends(get_memory_service),
) -> Dict[str, Any]:
    import os, json, datetime
    base = memory_service._store_dir(ctx.tenant_id, ctx.project_id)
    sid = payload.get('session_id') or 'default'
    path = memory_service._session_file(base, sid)
    rec = {
        'session_id': sid,
        'content': payload.get('content') or '',
        'project_id': ctx.project_id,
        'tenant': ctx.tenant_id,
        'ts': payload.get('ts') or datetime.datetime.utcnow().isoformat(),
    }
    ttl = payload.get('ttl_seconds')
    if ttl:
        rec['expires_at'] = (datetime.datetime.utcnow()+datetime.timedelta(seconds=int(ttl))).isoformat()
    if payload.get('expires_at'):
        rec['expires_at'] = payload['expires_at']
    memory_service._append_jsonl(path, rec)
    return {'status':'ok'}

@router.get("/session/list", status_code=status.HTTP_200_OK, response_model=None)
def session_list(
    session_id: str,
    limit: int = 100,
    offset: int = 0,
    since: Optional[str] = None,
    until: Optional[str] = None,
    ctx: AuthContext = Depends(require_auth_context),
    memory_service: MemoryService = Depends(get_memory_service),
) -> List[Dict[str, Any]]:
    import os, json, datetime
    base = memory_service._store_dir(ctx.tenant_id, ctx.project_id)
    path = memory_service._session_file(base, session_id)
    if not os.path.exists(path):
        legacy = os.path.join(base, f'session_{session_id}.jsonl')
        if os.path.exists(legacy):
            path = legacy
    out=[]
    def parse_ts(ts: str):
        try:
            return datetime.datetime.fromisoformat(ts.replace('Z','+00:00'))
        except Exception:
            return None
    t_since = parse_ts(since) if since else None
    t_until = parse_ts(until) if until else None
    if os.path.exists(path):
        with open(path,'r',encoding='utf-8') as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    exp = obj.get('expires_at')
                    if exp:
                        try:
                            if datetime.datetime.fromisoformat(exp) < datetime.datetime.utcnow():
                                continue
                        except Exception:
                            pass
                    ts = parse_ts(obj.get('ts',''))
                    if t_since and ts and ts < t_since:
                        continue
                    if t_until and ts and ts > t_until:
                        continue
                    out.append(obj)
                except Exception:
                    pass
    return out[offset:offset+limit]

@router.post("/working/add", status_code=status.HTTP_201_CREATED, response_model=None)
def working_add(
    payload: Dict[str, Any] = Body(...),
    ctx: AuthContext = Depends(require_auth_context),
    memory_service: MemoryService = Depends(get_memory_service),
) -> Dict[str, Any]:
    import os, json, datetime
    base = memory_service._store_dir(ctx.tenant_id, ctx.project_id)
    path = memory_service._working_file(base)
    rec = {
        'content': payload.get('content') or '',
        'project_id': ctx.project_id,
        'tenant': ctx.tenant_id,
        'ts': payload.get('ts') or datetime.datetime.utcnow().isoformat(),
    }
    ttl = payload.get('ttl_seconds')
    if ttl:
        rec['expires_at'] = (datetime.datetime.utcnow()+datetime.timedelta(seconds=int(ttl))).isoformat()
    if payload.get('expires_at'):
        rec['expires_at'] = payload['expires_at']
    memory_service._append_jsonl(path, rec)
    return {'status':'ok'}

@router.get("/working/list", status_code=status.HTTP_200_OK, response_model=None)
def working_list(
    limit: int = 100,
    offset: int = 0,
    since: Optional[str] = None,
    until: Optional[str] = None,
    ctx: AuthContext = Depends(require_auth_context),
    memory_service: MemoryService = Depends(get_memory_service),
) -> List[Dict[str, Any]]:
    import os, json, datetime
    base = memory_service._store_dir(ctx.tenant_id, ctx.project_id)
    path = memory_service._working_file(base)
    out=[]
    def parse_ts(ts: str):
        try:
            return datetime.datetime.fromisoformat(ts.replace('Z','+00:00'))
        except Exception:
            return None
    t_since = parse_ts(since) if since else None
    t_until = parse_ts(until) if until else None
    if os.path.exists(path):
        with open(path,'r',encoding='utf-8') as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    exp = obj.get('expires_at')
                    if exp:
                        try:
                            if datetime.datetime.fromisoformat(exp) < datetime.datetime.utcnow():
                                continue
                        except Exception:
                            pass
                    ts = parse_ts(obj.get('ts',''))
                    if t_since and ts and ts < t_since:
                        continue
                    if t_until and ts and ts > t_until:
                        continue
                    out.append(obj)
                except Exception:
                    pass
    return out[offset:offset+limit]
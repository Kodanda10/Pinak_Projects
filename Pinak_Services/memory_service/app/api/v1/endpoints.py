from fastapi import APIRouter, status, Header, HTTPException, Body, Depends
import os
import secrets
import jwt
import datetime
from app.core.schemas import MemoryCreate, MemoryRead, MemorySearchResult
from app.services.memory_service import (
    MemoryService,
    add_episodic as svc_add_episodic,
    list_episodic as svc_list_episodic,
    add_procedural as svc_add_procedural,
    list_procedural as svc_list_procedural,
    add_rag as svc_add_rag,
    list_rag as svc_list_rag,
    search_v2 as svc_search_v2,
)

def get_memory_service():
    """Dependency that provides a MemoryService instance."""
    return MemoryService()

from typing import List, Dict, Any, Optional, Union

router = APIRouter()
_SECRET_KEY = os.getenv('SECRET_KEY') or secrets.token_urlsafe(32)

@router.post("/add", response_model=MemoryRead, status_code=status.HTTP_201_CREATED)
def add_memory(memory: MemoryCreate, memory_service: MemoryService = Depends(get_memory_service)):
    """API endpoint to add a new memory."""
    return memory_service.add_memory(memory)

@router.get("/search", response_model=List[MemorySearchResult])
def search_memory(query: str, k: int = 5, memory_service: MemoryService = Depends(get_memory_service)):
    """API endpoint to search for relevant memories."""
    return memory_service.search_memory(query=query, k=k)

@router.post("/episodic/add", status_code=status.HTTP_201_CREATED)
def add_episodic(payload: Dict[str, Any] = Body(...), project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project"), memory_service: MemoryService = Depends(get_memory_service)) -> Dict[str, Any]:
    tenant = "default"  # Simplified
    rec = svc_add_episodic(memory_service, tenant, project_id or "default", payload.get('content') or '', int(payload.get('salience') or 0))
    return rec

@router.get("/episodic/list", status_code=status.HTTP_200_OK)
def list_episodic(project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project"), memory_service: MemoryService = Depends(get_memory_service)) -> list[Dict[str, Any]]:
    tenant = "default"  # Simplified
    return svc_list_episodic(memory_service, tenant, project_id or "default")

@router.post("/procedural/add", status_code=status.HTTP_201_CREATED)
def add_procedural(payload: Dict[str, Any] = Body(...), project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project"), memory_service: MemoryService = Depends(get_memory_service)) -> Dict[str, Any]:
    tenant = "default"  # Simplified
    rec = svc_add_procedural(memory_service, tenant, project_id or "default", payload.get('skill_id') or 'skill', payload.get('steps') or [])
    return rec

@router.get("/procedural/list", status_code=status.HTTP_200_OK)
def list_procedural(project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project"), memory_service: MemoryService = Depends(get_memory_service)) -> list[Dict[str, Any]]:
    tenant = "default"  # Simplified
    return svc_list_procedural(memory_service, tenant, project_id or "default")

@router.post("/rag/add", status_code=status.HTTP_201_CREATED)
def add_rag(payload: Dict[str, Any] = Body(...), project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project"), memory_service: MemoryService = Depends(get_memory_service)) -> Dict[str, Any]:
    tenant = "default"  # Simplified
    rec = svc_add_rag(memory_service, tenant, project_id or "default", payload.get('query') or '', payload.get('external_source') or '')
    return rec

@router.get("/rag/list", status_code=status.HTTP_200_OK)
def list_rag(project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project"), memory_service: MemoryService = Depends(get_memory_service)) -> list[Dict[str, Any]]:
    tenant = "default"  # Simplified
    return svc_list_rag(memory_service, tenant, project_id or "default")

@router.get("/search_v2", status_code=status.HTTP_200_OK)
def search_v2(query: str, layers: str = 'episodic', limit: int = 20, offset: int = 0, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project"), memory_service: MemoryService = Depends(get_memory_service)) -> Dict[str, Any]:
    tenant = "default"  # Simplified
    layer_list = [s.strip() for s in layers.split(',') if s.strip()]
    res = svc_search_v2(memory_service, tenant, project_id or "default", query, layer_list)
    for k,v in list(res.items()):
        if isinstance(v, list):
            res[k] = v[offset:offset+limit]
    return res

@router.post("/event", status_code=status.HTTP_201_CREATED, response_model=None)
def add_event(payload: Dict[str, Any] = Body(...), project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project"), memory_service: MemoryService = Depends(get_memory_service)) -> Dict[str, Any]:
    tenant = payload.get("tenant", "default")
    base = memory_service._store_dir(tenant, project_id or "default")
    ev = {"ts": datetime.datetime.utcnow().isoformat(), **payload, "project_id": project_id}
    ep = memory_service._dated_file(base, 'events', 'events')
    memory_service._append_audit_jsonl(ep, ev)
    return {"status":"ok"}

@router.get("/events", status_code=status.HTTP_200_OK, response_model=None)
def list_events(q: Optional[str] = None, since: Optional[str] = None, until: Optional[str] = None, limit: int = 100, offset: int = 0, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project"), memory_service: MemoryService = Depends(get_memory_service)) -> List[Dict[str, Any]]:
    import json, os, datetime
    tenant = "default"
    base = memory_service._store_dir(tenant, project_id or "default")
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
def session_add(payload: Dict[str, Any] = Body(...), project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project"), memory_service: MemoryService = Depends(get_memory_service)) -> Dict[str, Any]:
    import os, json, datetime
    tenant = payload.get("tenant", "default")
    base = memory_service._store_dir(tenant, project_id or "default")
    sid = payload.get('session_id') or 'default'
    path = memory_service._session_file(base, sid)
    rec = {
        'session_id': sid,
        'content': payload.get('content') or '',
        'project_id': project_id,
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
def session_list(session_id: str, limit: int = 100, offset: int = 0, since: Optional[str] = None, until: Optional[str] = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project"), memory_service: MemoryService = Depends(get_memory_service)) -> List[Dict[str, Any]]:
    import os, json, datetime
    tenant = "default"
    base = memory_service._store_dir(tenant, project_id or "default")
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
def working_add(payload: Dict[str, Any] = Body(...), project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project"), memory_service: MemoryService = Depends(get_memory_service)) -> Dict[str, Any]:
    import os, json, datetime
    tenant = payload.get("tenant", "default")
    base = memory_service._store_dir(tenant, project_id or "default")
    path = memory_service._working_file(base)
    rec = {
        'content': payload.get('content') or '',
        'project_id': project_id,
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
def working_list(limit: int = 100, offset: int = 0, since: Optional[str] = None, until: Optional[str] = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project"), memory_service: MemoryService = Depends(get_memory_service)) -> List[Dict[str, Any]]:
    import os, json, datetime
    tenant = "default"
    base = memory_service._store_dir(tenant, project_id or "default")
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
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
def search_v2(query: str, layers: str = 'semantic', limit: int = 20, offset: int = 0, request: Request = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> Dict[str, Any]:
    tenant = resolve_tenant(request, {}) if request is not None else "default"
    layer_list = [s.strip() for s in layers.split(',') if s.strip()]
    res = svc_search_v2(memory_service, tenant, project_id, query, layer_list)
    for k,v in list(res.items()):
        if isinstance(v, list):
            res[k] = v[offset:offset+limit]
    return res

@router.post("/changelog/redact", status_code=status.HTTP_200_OK)
def redact(memory_id: str = Body(..., embed=True), reason: str = Body('redact', embed=True), request: Request = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> Dict[str, Any]:
    tenant = resolve_tenant(request, {}) if request is not None else "default"
    return memory_service.redact_memory(memory_id, tenant, project_id, reason)

@router.get("/changelog", status_code=status.HTTP_200_OK)
def list_changelog(change_type: Optional[str] = None, since: Optional[str] = None, until: Optional[str] = None, limit: int = 100, offset: int = 0, request: Request = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> List[Dict[str, Any]]:
    tenant = resolve_tenant(request, {}) if request is not None else "default"
    return memory_service.list_changelog(tenant, project_id, change_type, since, until, limit, offset)

@router.post("/event", status_code=status.HTTP_201_CREATED)
def add_event(payload: Dict[str, Any] = Body(...), request: Request = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> Dict[str, Any]:
    tenant = resolve_tenant(request, payload) if request is not None else payload.get("tenant", "default")
    base = memory_service._store_dir(tenant, project_id)
    import datetime, os, json
    ev = {"ts": datetime.datetime.utcnow().isoformat(), **payload, "project_id": project_id}
    memory_service._append_jsonl(os.path.join(base, 'events.jsonl'), ev)
    # If this is a governance audit event, mirror into changelog as change_type=governance
    try:
        if payload.get('type') == 'gov_audit':
            memory_service._append_jsonl(os.path.join(base, 'changelog.jsonl'), {
                'change_type': 'governance',
                'ts': ev['ts'],
                'project_id': project_id,
                'path': payload.get('path'),
                'method': payload.get('method'),
                'status': payload.get('status'),
                'role': payload.get('role')
            })
    except Exception:
        pass
    return {"status":"ok"}

@router.get("/events", status_code=status.HTTP_200_OK)
def list_events(q: Optional[str] = None, since: Optional[str] = None, until: Optional[str] = None, limit: int = 100, offset: int = 0, request: Request = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> List[Dict[str, Any]]:
    import json, os, datetime
    tenant = resolve_tenant(request, {}) if request is not None else "default"
    base = memory_service._store_dir(tenant, project_id)
    p = os.path.join(base, 'events.jsonl')
    out=[]
    def parse_ts(ts: str):
        try:
            # allow Z suffix
            return datetime.datetime.fromisoformat(ts.replace('Z','+00:00'))
        except Exception:
            return None
    t_since = parse_ts(since) if since else None
    t_until = parse_ts(until) if until else None
    if os.path.exists(p):
        with open(p,'r',encoding='utf-8') as fh:
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

# --- Session endpoints (persisted JSONL with TTL support) ---
@router.post("/session/add", status_code=status.HTTP_201_CREATED)
def session_add(payload: Dict[str, Any] = Body(...), request: Request = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> Dict[str, Any]:
    import os, json, datetime
    tenant = resolve_tenant(request, payload) if request is not None else payload.get("tenant", "default")
    base = memory_service._store_dir(tenant, project_id)
    sid = payload.get('session_id') or 'default'
    path = os.path.join(base, f'session_{sid}.jsonl')
    rec = {
        'session_id': sid,
        'content': payload.get('content') or '',
        'project_id': project_id,
        'ts': payload.get('ts') or datetime.datetime.utcnow().isoformat(),
    }
    ttl = payload.get('ttl_seconds')
    if ttl:
        rec['expires_at'] = (datetime.datetime.utcnow()+datetime.timedelta(seconds=int(ttl))).isoformat()
    # allow explicit expires_at for testing
    if payload.get('expires_at'):
        rec['expires_at'] = payload['expires_at']
    memory_service._append_jsonl(path, rec)
    return {'status':'ok'}

@router.get("/session/list", status_code=status.HTTP_200_OK)
def session_list(session_id: str, limit: int = 100, offset: int = 0, since: Optional[str] = None, until: Optional[str] = None, request: Request = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> List[Dict[str, Any]]:
    import os, json, datetime
    tenant = resolve_tenant(request, {}) if request is not None else "default"
    base = memory_service._store_dir(tenant, project_id)
    path = os.path.join(base, f'session_{session_id}.jsonl')
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
                    # filter expired
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

# --- Working endpoints (persisted JSONL with TTL support) ---
@router.post("/working/add", status_code=status.HTTP_201_CREATED)
def working_add(payload: Dict[str, Any] = Body(...), request: Request = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> Dict[str, Any]:
    import os, json, datetime
    tenant = resolve_tenant(request, payload) if request is not None else payload.get("tenant", "default")
    base = memory_service._store_dir(tenant, project_id)
    path = os.path.join(base, 'working.jsonl')
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

@router.get("/working/list", status_code=status.HTTP_200_OK)
def working_list(limit: int = 100, offset: int = 0, since: Optional[str] = None, until: Optional[str] = None, request: Request = None, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> List[Dict[str, Any]]:
    import os, json, datetime
    tenant = resolve_tenant(request, {}) if request is not None else "default"
    base = memory_service._store_dir(tenant, project_id)
    path = os.path.join(base, 'working.jsonl')
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

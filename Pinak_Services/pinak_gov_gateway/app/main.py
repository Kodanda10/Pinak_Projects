from __future__ import annotations

import os
from typing import Optional

import httpx
import json
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, Request, Response, Header, HTTPException
from fastapi.responses import JSONResponse
import jwt
import secrets


GOV_UPSTREAM = os.getenv("GOV_UPSTREAM", "http://parlant:8800")
GOV_UPSTREAM_CA = os.getenv("GOV_UPSTREAM_CA")  # optional CA bundle for TLS pinning
MEMORY_API_URL = os.getenv("MEMORY_API_URL", "http://memory-api:8000")
MEMORY_API_CA = os.getenv("MEMORY_API_CA")  # optional CA bundle for TLS pinning
MEMORY_API_CLIENT_CERT = os.getenv("MEMORY_API_CLIENT_CERT")  # optional client cert (mTLS)
MEMORY_API_CLIENT_KEY = os.getenv("MEMORY_API_CLIENT_KEY")    # optional client key (mTLS)
OPA_URL = os.getenv("OPA_URL")  # optional OPA for policy checks
GOV_AUDIT_DIR = os.getenv("GOV_AUDIT_DIR", "/data/gov")
# Do not hardcode secrets; prefer env. Generate ephemeral dev secret if unset.
SECRET_KEY = os.getenv("SECRET_KEY") or secrets.token_urlsafe(32)
REQUIRE_PROJECT_HEADER = os.getenv("REQUIRE_PROJECT_HEADER", "true").lower() in {"1","true","yes","on"}
# Allowed roles for RBAC propagation; empty means don't enforce specific roles
ALLOWED_ROLES = {r.strip() for r in os.getenv("PINAK_ALLOWED_ROLES", "viewer,editor,admin").split(',') if r.strip()}

app = FastAPI(title="Pinak-Gov Gateway")


def enforce_project_and_pid(request: Request, project_id: Optional[str]) -> dict:
    if REQUIRE_PROJECT_HEADER and not project_id:
        raise HTTPException(status_code=400, detail="Missing X-Pinak-Project header")
    auth = request.headers.get("Authorization")
    out_claims: dict = {}
    if auth and auth.lower().startswith("bearer ") and project_id:
        token = auth.split(" ", 1)[1]
        try:
            claims = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            pid = claims.get("pid")
            if pid and pid != project_id:
                raise HTTPException(status_code=403, detail="Project header/token mismatch")
            # Optional RBAC propagation
            role = claims.get("role") or claims.get("roles")
            if isinstance(role, list) and role:
                role = role[0]
            if role:
                out_claims["role"] = str(role)
                if ALLOWED_ROLES and role not in ALLOWED_ROLES:
                    raise HTTPException(status_code=403, detail="Role not permitted")
            out_claims["pid"] = pid
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")
    return out_claims


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "upstream": GOV_UPSTREAM}


@app.api_route("/{path:path}", methods=["GET","POST","PUT","PATCH","DELETE"])
async def proxy(path: str, request: Request, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> Response:
    # Enforce Bridge project + pid claim
    claims = enforce_project_and_pid(request, project_id)

    url = f"{GOV_UPSTREAM}/{path}".rstrip("/")
    # Build upstream request
    headers = dict(request.headers)
    # Ensure X-Pinak-Project is forwarded
    if project_id:
        headers['X-Pinak-Project'] = project_id
    # Forward role to upstream if present
    role = claims.get("role") if isinstance(claims, dict) else None
    if role:
        headers['X-Pinak-Role'] = role

    body = await request.body()
    method = request.method.upper()
    verify_param = GOV_UPSTREAM_CA if GOV_UPSTREAM_CA else True
    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0, verify=verify_param) as client:
        # Optional OPA/Rego policy check
        if OPA_URL:
            try:
                opa_in = {
                    "claims": claims,
                    "headers": {k.lower(): v for k, v in headers.items()},
                    "request": {"path": f"/{path}", "method": method},
                }
                r = await client.post(f"{OPA_URL.rstrip('/')}/v1/data/pinak/policy/allow", json={"input": opa_in}, timeout=5.0)
                if r.status_code == 200:
                    data = r.json()
                    decision = data.get("result")
                    if decision is False:
                        raise HTTPException(status_code=403, detail="Denied by policy")
            except HTTPException:
                raise
            except Exception:
                # Fail-open if OPA is unavailable
                pass
        upstream = await client.request(method, url, headers=headers, content=body)
    # Best-effort governance audit sync for mutating ops
    try:
        if method in {"POST","PUT","PATCH","DELETE"} and project_id and upstream.status_code < 400:
            # Write to per-tenant/project audit under /data/gov
            base = Path(GOV_AUDIT_DIR) / (project_id or "default")
            base.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "project_id": project_id,
                "method": method,
                "path": f"/{path}",
                "status": upstream.status_code,
            }
            try:
                if body:
                    entry["request"] = json.loads(body.decode("utf-8"))
            except Exception:
                pass
            (base / "audit.jsonl").write_text(((base / "audit.jsonl").read_text() if (base/"audit.jsonl").exists() else "") + json.dumps(entry) + "\n")
            # Also sync into memory events for unified audit
            try:
                verify_mem = MEMORY_API_CA if MEMORY_API_CA else True
                cert_opt = (MEMORY_API_CLIENT_CERT, MEMORY_API_CLIENT_KEY) if MEMORY_API_CLIENT_CERT and MEMORY_API_CLIENT_KEY else None
                async with httpx.AsyncClient(timeout=5.0, verify=verify_mem, cert=cert_opt) as client:
                    await client.post(f"{MEMORY_API_URL}/api/v1/memory/event", json={
                        "type": "gov_audit",
                        "path": f"/{path}",
                        "method": method,
                        "status": upstream.status_code,
                        "role": role or None,
                    }, headers={'X-Pinak-Project': project_id})
            except Exception:
                pass
    except Exception:
        pass
    return Response(content=upstream.content, status_code=upstream.status_code, headers=dict(upstream.headers), media_type=upstream.headers.get('content-type'))

@app.get("/pinak-gov/audit")
async def audit_list(project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project"), since: Optional[str] = None, until: Optional[str] = None, limit: int = 100, offset: int = 0):
    enforce_project_and_pid(Request({'type':'http'}), project_id)
    base = Path("/data/gov") / (project_id or "default")
    p = base / "audit.jsonl"
    out = []
    def parse(ts: str):
        try:
            return datetime.fromisoformat(ts.replace('Z','+00:00'))
        except Exception:
            return None
    t_since = parse(since) if since else None
    t_until = parse(until) if until else None
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                obj = json.loads(line)
                ts = parse(obj.get('ts',''))
                if t_since and ts and ts < t_since:
                    continue
                if t_until and ts and ts > t_until:
                    continue
                out.append(obj)
            except Exception:
                pass
    return JSONResponse(content=out[offset:offset+limit])

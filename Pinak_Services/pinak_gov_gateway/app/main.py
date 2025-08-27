from __future__ import annotations

import os
from typing import Optional

import httpx
import json
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, Request, Response, Header, HTTPException
from fastapi.responses import JSONResponse
from jose import jwt, JWTError


GOV_UPSTREAM = os.getenv("GOV_UPSTREAM", "http://parlant:8800")
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-prod")
REQUIRE_PROJECT_HEADER = os.getenv("REQUIRE_PROJECT_HEADER", "true").lower() in {"1","true","yes","on"}

app = FastAPI(title="Pinak-Gov Gateway")


def enforce_project_and_pid(request: Request, project_id: Optional[str]) -> None:
    if REQUIRE_PROJECT_HEADER and not project_id:
        raise HTTPException(status_code=400, detail="Missing X-Pinak-Project header")
    auth = request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer ") and project_id:
        token = auth.split(" ", 1)[1]
        try:
            claims = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
            pid = claims.get("pid")
            if pid and pid != project_id:
                raise HTTPException(status_code=403, detail="Project header/token mismatch")
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/health")
async def health() -> dict:
    return {"ok": True, "upstream": GOV_UPSTREAM}


@app.api_route("/{path:path}", methods=["GET","POST","PUT","PATCH","DELETE"])
async def proxy(path: str, request: Request, project_id: Optional[str] = Header(default=None, alias="X-Pinak-Project")) -> Response:
    # Enforce Bridge project + pid claim
    enforce_project_and_pid(request, project_id)

    url = f"{GOV_UPSTREAM}/{path}".rstrip("/")
    # Build upstream request
    headers = dict(request.headers)
    # Ensure X-Pinak-Project is forwarded
    if project_id:
        headers['X-Pinak-Project'] = project_id

    body = await request.body()
    method = request.method.upper()
    async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
        upstream = await client.request(method, url, headers=headers, content=body)
    # Best-effort governance audit sync for mutating ops
    try:
        if method in {"POST","PUT","PATCH","DELETE"} and project_id and upstream.status_code < 400:
            # Write to per-tenant/project audit under /data/gov
            base = Path("/data/gov") / (project_id or "default")
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

from __future__ import annotations

import os
from typing import Optional

import httpx
from fastapi import FastAPI, Request, Response, Header, HTTPException
from fastapi.responses import JSONResponse
from jose import jwt, JWTError


GOV_UPSTREAM = os.getenv("GOV_UPSTREAM", "http://parlant:8800")
SECRET_KEY = os.getenv("SECRET_KEY", "change-me-in-prod")
REQUIRE_PROJECT_HEADER = os.getenv("REQUIRE_PROJECT_HEADER", "true").lower() in {"1","true","yes","on"}

app = FastAPI(title="Pinak Governance Gateway")


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
    return Response(content=upstream.content, status_code=upstream.status_code, headers=dict(upstream.headers), media_type=upstream.headers.get('content-type'))


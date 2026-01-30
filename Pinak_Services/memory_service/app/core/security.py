"""Security utilities for enforcing JWT-based authentication."""

from dataclasses import dataclass
from typing import List, Optional
import os

import jwt
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


@dataclass
class AuthContext:
    """Represents the authenticated request context."""

    subject: Optional[str]
    tenant_id: str
    project_id: str
    roles: List[str]
    scopes: List[str]
    client_name: Optional[str]
    client_id: Optional[str]
    parent_client_id: Optional[str]
    child_client_id: Optional[str]
    effective_client_id: str
    token: str


_http_bearer = HTTPBearer(auto_error=False)


def _get_secret() -> str:
    secret = os.getenv("PINAK_JWT_SECRET")
    if not secret:
        raise RuntimeError("PINAK_JWT_SECRET environment variable must be set")
    return secret


def require_auth_context(
    credentials: HTTPAuthorizationCredentials = Depends(_http_bearer),
    child_client_id: Optional[str] = Header(default=None, alias="X-Pinak-Child-Id"),
) -> AuthContext:
    """FastAPI dependency that validates a JWT and extracts tenant context."""

    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token = credentials.credentials
    secret = _get_secret()
    algorithm = os.getenv("PINAK_JWT_ALGORITHM", "HS256")


    try:
        payload = jwt.decode(token, secret, algorithms=[algorithm])
    except jwt.ExpiredSignatureError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired") from exc
    except jwt.InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    tenant = payload.get("tenant") or payload.get("tenant_id")
    project_id = payload.get("project_id") or payload.get("project")
    if not tenant or not project_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Tenant or project missing in token")

    roles = payload.get("roles") or []
    if not isinstance(roles, list):
        roles = [str(roles)]

    client_name = payload.get("client_name") or payload.get("client")
    client_id = payload.get("client_id") or payload.get("cid") or client_name
    parent_client_id = payload.get("parent_client_id") or payload.get("parent_client")
    effective_client_id = child_client_id or client_id or payload.get("sub") or "unknown"

    return AuthContext(
        subject=payload.get("sub"),
        tenant_id=str(tenant),
        project_id=str(project_id),
        roles=[str(role) for role in roles],
        scopes=[str(scope) for scope in (payload.get("scopes") or [])],
        client_name=client_name,
        client_id=client_id,
        parent_client_id=parent_client_id,
        child_client_id=child_client_id,
        effective_client_id=effective_client_id,
        token=token,
    )


def require_scope(ctx: AuthContext, scope: str) -> None:
    enforce = os.getenv("PINAK_ENFORCE_SCOPES", "true").lower() in ("1", "true", "yes")
    if not enforce:
        return
    if scope not in ctx.scopes:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Missing scope: {scope}")


def require_role(ctx: AuthContext, role: str) -> None:
    enforce = os.getenv("PINAK_ENFORCE_SCOPES", "true").lower() in ("1", "true", "yes")
    if not enforce:
        return
    if role not in ctx.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"Missing role: {role}")

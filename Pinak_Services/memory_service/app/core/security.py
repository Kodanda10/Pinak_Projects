"""Security utilities for enforcing JWT-based authentication."""

from dataclasses import dataclass
from typing import List, Optional
import os

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


@dataclass
class AuthContext:
    """Represents the authenticated request context."""

    subject: Optional[str]
    tenant_id: str
    project_id: str
    roles: List[str]
    token: str


_http_bearer = HTTPBearer(auto_error=False)


def _get_secret() -> str:
    secret = os.getenv("PINAK_JWT_SECRET")
    if not secret:
        raise RuntimeError("PINAK_JWT_SECRET environment variable must be set")
    return secret


def require_auth_context(
    credentials: HTTPAuthorizationCredentials = Depends(_http_bearer),
) -> AuthContext:
    """FastAPI dependency that validates a JWT and extracts tenant context."""

    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")

    token = credentials.credentials
    secret = _get_secret()
    algorithm = os.getenv("PINAK_JWT_ALGORITHM", "HS256")

    try:
        payload = jwt.decode(token, secret, algorithms=[algorithm], options={"require": ["tenant", "project_id"]})
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

    return AuthContext(
        subject=payload.get("sub"),
        tenant_id=str(tenant),
        project_id=str(project_id),
        roles=[str(role) for role in roles],
        token=token,
    )

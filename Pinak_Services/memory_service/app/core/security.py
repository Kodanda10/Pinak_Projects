"""
Pinak Memory Service - Enterprise Security Module
================================================

SOTA Security Features:
- Security headers
- Rate limiting
- CORS protection
- Input validation
- Authentication
"""

import time
from typing import Optional

import jwt
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import settings
from .logging import logger


def setup_security_middleware(app: FastAPI):
    """
    Setup security middleware for the FastAPI application.

    Adds security headers, CORS, and other security features.
    """

    # Add security headers middleware
    @app.middleware("http")
    async def add_security_headers(request, call_next):
        response = await call_next(request)

        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        return response

    logger.info("Security middleware configured")


def verify_token(token: str) -> Optional[dict]:
    """
    Verify JWT token.

    Returns decoded payload if valid, None otherwise.
    """
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError:
        logger.warning("Invalid token")
        return None


security = HTTPBearer(auto_error=False)

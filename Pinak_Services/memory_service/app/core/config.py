"""
Pinak Memory Service - Enterprise Configuration Management
==========================================================

SOTA Configuration Management with:
- Environment-based configuration
- Validation and type safety
- Security-first defaults
- Production-ready settings
- Feature flags and toggles
"""

import os
import secrets
from typing import List, Optional, Set

from pydantic import BaseModel, Field, validator


class Settings(BaseModel):
    """
    Enterprise-grade configuration management using Pydantic.

    Features:
    - Environment variable binding
    - Validation and type coercion
    - Security-first defaults
    - Production-ready settings
    """

    # Application
    APP_NAME: str = "Pinak Memory Service"
    VERSION: str = "1.3.1"
    DEBUG: bool = Field(default=False)
    PRODUCTION: bool = Field(default=False)

    # Server
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    WORKERS: int = Field(default=1)

    # Security
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    JWT_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # CORS
    ALLOWED_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"]
    )
    ALLOWED_HOSTS: List[str] = Field(default=["*"])

    # Database
    DATABASE_URL: str = Field(default="sqlite:///./data/memory.db")
    DB_POOL_SIZE: int = Field(default=10)
    DB_MAX_OVERFLOW: int = Field(default=20)
    DB_POOL_RECYCLE: int = Field(default=3600)

    # Vector Store
    VECTOR_DB_PATH: str = Field(default="./data/memory.faiss")
    VECTOR_DIMENSION: int = Field(default=384)
    VECTOR_INDEX_TYPE: str = Field(default="IndexFlatIP")

    # Redis
    REDIS_URL: Optional[str] = Field(default=None)
    REDIS_POOL_SIZE: int = Field(default=10)
    REDIS_DB: int = Field(default=0)

    # Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(default=1000)
    RATE_LIMIT_BURST: int = Field(default=100)

    # Circuit Breaker
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(default=5)
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(default=60)
    CIRCUIT_BREAKER_SUCCESS_THRESHOLD: int = Field(default=3)

    # Observability
    METRICS_ENABLED: bool = Field(default=True)
    TRACING_ENABLED: bool = Field(default=False)
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="json")

    # Feature Flags
    ENABLE_ADVANCED_SEARCH: bool = Field(default=True)
    ENABLE_CROSS_LAYER_SEARCH: bool = Field(default=True)
    ENABLE_SEMANTIC_CACHE: bool = Field(default=True)
    ENABLE_RATE_LIMITING: bool = Field(default=True)
    ENABLE_CIRCUIT_BREAKER: bool = Field(default=True)

    # Security
    FAIL_FAST_ON_STARTUP: bool = Field(default=True)
    ENCRYPT_SENSITIVE_DATA: bool = Field(default=True)
    ENABLE_AUDIT_LOGGING: bool = Field(default=True)

    # Performance
    MAX_REQUEST_SIZE: int = Field(default=10 * 1024 * 1024)  # 10MB
    REQUEST_TIMEOUT: int = Field(default=30)
    CONNECTION_TIMEOUT: int = Field(default=10)

    # Data Management
    DATA_RETENTION_DAYS: int = Field(default=365)
    BACKUP_ENABLED: bool = Field(default=True)
    BACKUP_INTERVAL_HOURS: int = Field(default=24)

    @validator("ALLOWED_ORIGINS", pre=True)
    def parse_allowed_origins(cls, v):
        """Parse comma-separated origins."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("ALLOWED_HOSTS", pre=True)
    def parse_allowed_hosts(cls, v):
        """Parse comma-separated hosts."""
        if isinstance(v, str):
            return [host.strip() for host in v.split(",")]
        return v

    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        """Validate database URL format."""
        if not v:
            raise ValueError("DATABASE_URL is required")
        return v

    @validator("SECRET_KEY", "JWT_SECRET_KEY")
    def validate_secret_keys(cls, v):
        """Ensure secret keys are sufficiently long."""
        if len(v) < 32:
            raise ValueError("Secret keys must be at least 32 characters long")
        return v

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Global settings instance
settings = Settings()

# Environment-specific overrides
if os.getenv("PRODUCTION", "false").lower() in ("true", "1", "yes"):
    # Production hardening
    settings.DEBUG = False
    settings.LOG_LEVEL = "WARNING"
    settings.METRICS_ENABLED = True
    settings.TRACING_ENABLED = True
    settings.ENABLE_RATE_LIMITING = True
    settings.ENABLE_CIRCUIT_BREAKER = True
    settings.FAIL_FAST_ON_STARTUP = True
    settings.PRODUCTION = True

elif os.getenv("DEBUG", "false").lower() in ("true", "1", "yes"):
    # Development settings
    settings.LOG_LEVEL = "DEBUG"
    settings.METRICS_ENABLED = False
    settings.TRACING_ENABLED = False
    settings.ENABLE_RATE_LIMITING = False
    settings.ENABLE_CIRCUIT_BREAKER = False
    settings.FAIL_FAST_ON_STARTUP = False
    settings.DEBUG = True

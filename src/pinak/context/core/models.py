# FANG-Level Context Orchestrator - Core Data Models
"""
Enterprise-grade data models for Pinakontext SOTA Context Orchestrator.
Implements comprehensive type safety, validation, and serialization.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Protocol, Union

from pydantic import BaseModel, Field, validator


class ContextLayer(str, Enum):
    """Memory layers available for context retrieval."""

    SEMANTIC = "semantic"
    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    RAG = "rag"
    EVENTS = "events"
    SESSION = "session"
    WORKING = "working"
    CHANGELOG = "changelog"


class ContextPriority(str, Enum):
    """Priority levels for context items."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SecurityClassification(str, Enum):
    """Security classification levels for context data."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ContextItem(BaseModel):
    """Enterprise-grade context item with comprehensive metadata."""

    # Core identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(..., min_length=1, max_length=200)
    summary: str = Field(..., min_length=1, max_length=1000)

    # Content and metadata
    content: str = Field(..., min_length=1)
    layer: ContextLayer
    priority: ContextPriority = ContextPriority.MEDIUM
    classification: SecurityClassification = SecurityClassification.INTERNAL

    # Temporal information
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None

    # Relationships and references
    project_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)
    parent_id: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    references: List[str] = Field(default_factory=list)

    # Actions and recommendations
    actions: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

    # Quality and scoring
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.5)
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.5)
    freshness_score: float = Field(ge=0.0, le=1.0, default=1.0)

    # Security and audit
    created_by: str
    checksum: Optional[str] = None
    signature: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration for enterprise features."""

        validate_assignment = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    @validator("checksum", always=True)
    def generate_checksum(cls, v, values):
        """Generate tamper-evident checksum for the context item."""
        if v is not None:
            return v

        # Create canonical representation for hashing
        canonical_data = {
            "id": values.get("id", ""),
            "title": values.get("title", ""),
            "content": values.get("content", ""),
            "created_at": (
                values.get("created_at", "").isoformat()
                if values.get("created_at")
                else ""
            ),
            "project_id": values.get("project_id", ""),
            "tenant_id": values.get("tenant_id", ""),
        }

        canonical_json = json.dumps(canonical_data, sort_keys=True)
        return hashlib.sha256(canonical_json.encode()).hexdigest()

    def is_expired(self) -> bool:
        """Check if the context item has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def is_fresh(self, threshold_hours: int = 24) -> bool:
        """Check if the context item is fresh within the given threshold."""
        age = datetime.now(timezone.utc) - self.created_at
        return age.total_seconds() < (threshold_hours * 3600)

    def to_secure_dict(self, user_clearance: SecurityClassification) -> Dict[str, Any]:
        """Return a security-filtered dictionary representation."""
        if user_clearance.value < self.classification.value:
            # Return redacted version for insufficient clearance
            return {
                "id": self.id,
                "title": "[REDACTED]",
                "summary": "[REDACTED - Insufficient Security Clearance]",
                "classification": self.classification.value,
                "created_at": self.created_at.isoformat(),
            }

        return self.dict()

    def calculate_freshness_score(self) -> float:
        """Calculate freshness score based on age and updates."""
        now = datetime.now(timezone.utc)
        age_hours = (now - self.created_at).total_seconds() / 3600
        update_age_hours = (now - self.updated_at).total_seconds() / 3600

        # Exponential decay based on age (half-life of 24 hours)
        age_score = 0.5 ** (age_hours / 24)

        # Bonus for recent updates
        update_bonus = (
            0.2 if update_age_hours < 1 else 0.1 if update_age_hours < 24 else 0
        )

        return min(1.0, age_score + update_bonus)


class ContextQuery(BaseModel):
    """Enterprise-grade query model for context retrieval."""

    # Core identification
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Query content
    query: str = Field(..., min_length=1, max_length=1000)
    project_id: str = Field(..., min_length=1)
    tenant_id: str = Field(..., min_length=1)

    # Search parameters
    layers: List[ContextLayer] = Field(default_factory=lambda: list(ContextLayer))
    limit: int = Field(ge=1, le=100, default=20)
    offset: int = Field(ge=0, default=0)

    # Filtering and ranking
    min_relevance: float = Field(ge=0.0, le=1.0, default=0.1)
    min_confidence: float = Field(ge=0.0, le=1.0, default=0.1)
    priority_filter: Optional[ContextPriority] = None
    tags_filter: List[str] = Field(default_factory=list)

    # Temporal filters
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    include_expired: bool = False

    # Search options
    semantic_search: bool = True
    keyword_search: bool = True
    fuzzy_matching: bool = False

    # User context
    user_id: str
    user_clearance: SecurityClassification = SecurityClassification.INTERNAL

    # Performance tuning
    timeout_seconds: int = Field(ge=1, le=300, default=30)
    max_parallel_requests: int = Field(ge=1, le=10, default=3)


class ContextResponse(BaseModel):
    """Enterprise-grade response model for context queries."""

    # Query metadata
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    total_results: int = 0
    returned_results: int = 0
    execution_time_ms: int = 0

    # Results
    items: List[ContextItem] = Field(default_factory=list)

    # Performance metrics
    cache_hit: bool = False
    parallel_requests_used: int = 0

    # Security and audit
    filtered_count: int = 0
    redacted_count: int = 0
    audit_token: str = Field(default_factory=lambda: str(uuid.uuid4()))

    # Pagination
    has_more: bool = False
    next_offset: Optional[int] = None

    # Advanced metadata for world-beating retrieval
    metadata: Optional[Dict[str, Any]] = None

    def add_item(
        self, item: ContextItem, user_clearance: SecurityClassification
    ) -> None:
        """Add an item with security filtering."""
        if user_clearance.value >= item.classification.value:
            self.items.append(item)
            self.returned_results += 1
        else:
            self.redacted_count += 1

    def to_audit_log(self) -> Dict[str, Any]:
        """Generate audit log entry for this response."""
        return {
            "query_id": self.query_id,
            "audit_token": self.audit_token,
            "total_results": self.total_results,
            "returned_results": self.returned_results,
            "filtered_count": self.filtered_count,
            "redacted_count": self.redacted_count,
            "execution_time_ms": self.execution_time_ms,
            "cache_hit": self.cache_hit,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


class ContextEvent(BaseModel):
    """Event model for context system observability."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Event data
    data: Dict[str, Any] = Field(default_factory=dict)

    # Context
    project_id: str
    tenant_id: str
    user_id: Optional[str] = None

    # Metadata
    source: str = "pinakontext"
    version: str = "1.0.0"

    def to_log_entry(self) -> str:
        """Convert to structured log entry."""
        return json.dumps(
            {
                "event_id": self.event_id,
                "event_type": self.event_type,
                "timestamp": self.timestamp.isoformat(),
                "project_id": self.project_id,
                "tenant_id": self.tenant_id,
                "user_id": self.user_id,
                "source": self.source,
                "version": self.version,
                "data": self.data,
            },
            default=str,
        )


class IContextStore(Protocol):
    """Protocol for context storage implementations."""

    async def store(self, item: ContextItem) -> bool:
        """Store a context item."""
        ...

    async def retrieve(self, query: ContextQuery) -> ContextResponse:
        """Retrieve context items based on query."""
        ...

    async def delete(self, item_id: str, project_id: str) -> bool:
        """Delete a context item."""
        ...

    async def update(self, item: ContextItem) -> bool:
        """Update a context item."""
        ...

    async def search_similar(self, content: str, limit: int = 10) -> List[ContextItem]:
        """Search for similar context items."""
        ...

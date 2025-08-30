# FANG-Level Nudge Engine - Proactive Context Delivery
"""
Enterprise-grade Nudge Engine for Pinakontext SOTA Context Orchestrator.
Implements proactive context delivery with intelligent triggers and personalization.
"""


import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Union

from pydantic import BaseModel, Field, validator

from ..core.models import ContextItem, ContextLayer, ContextQuery, SecurityClassification

logger = logging.getLogger(__name__)


class NudgeType(str, Enum):
    """Types of proactive nudges."""

    CONTEXT_SUGGESTION = "context_suggestion"
    RELEVANT_UPDATE = "relevant_update"
    MISSING_CONTEXT = "missing_context"
    TIMING_OPTIMIZATION = "timing_optimization"
    SECURITY_ALERT = "security_alert"
    PERFORMANCE_TIP = "performance_tip"


class NudgePriority(str, Enum):
    """Priority levels for nudges."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NudgeTrigger(str, Enum):
    """Trigger conditions for nudges."""

    TIME_BASED = "time_based"
    CONTEXT_CHANGE = "context_change"
    USER_ACTIVITY = "user_activity"
    SYSTEM_EVENT = "system_event"
    PATTERN_RECOGNITION = "pattern_recognition"
    EXTERNAL_SIGNAL = "external_signal"


class NudgeCondition(BaseModel):
    """Condition for triggering a nudge."""

    trigger_type: NudgeTrigger
    parameters: Dict[str, Any] = Field(default_factory=dict)

    # Time-based conditions
    time_window_minutes: Optional[int] = None
    recurring_pattern: Optional[str] = None

    # Context-based conditions
    context_keywords: List[str] = Field(default_factory=list)
    layer_changes: List[ContextLayer] = Field(default_factory=list)

    # Activity-based conditions
    activity_patterns: List[str] = Field(default_factory=list)
    inactivity_threshold_minutes: Optional[int] = None

    # Pattern recognition
    pattern_confidence_threshold: float = Field(ge=0.0, le=1.0, default=0.7)


class NudgeTemplate(BaseModel):
    """Template for generating nudges."""

    template_id: str = Field(default_factory=lambda: f"template_{datetime.now().timestamp()}")
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)

    nudge_type: NudgeType
    priority: NudgePriority = NudgePriority.MEDIUM

    # Template content
    title_template: str
    message_template: str
    action_template: Optional[str] = None

    # Conditions for triggering
    conditions: List[NudgeCondition] = Field(default_factory=list)

    # Personalization
    personalization_rules: Dict[str, Any] = Field(default_factory=dict)

    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True

    # Usage tracking
    trigger_count: int = 0
    success_count: int = 0
    last_triggered: Optional[datetime] = None


class Nudge(BaseModel):
    """Proactive nudge for context delivery."""

    nudge_id: str = Field(default_factory=lambda: f"nudge_{datetime.now().timestamp()}")
    template_id: str

    # Target user/context
    user_id: str
    project_id: str
    tenant_id: str

    # Nudge content
    type: NudgeType
    priority: NudgePriority
    title: str
    message: str
    suggested_action: Optional[str] = None

    # Context information
    related_context_items: List[str] = Field(default_factory=list)
    context_query: Optional[ContextQuery] = None

    # Timing and delivery
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None

    # Personalization data
    personalization_data: Dict[str, Any] = Field(default_factory=dict)

    # Quality metrics
    relevance_score: float = Field(ge=0.0, le=1.0, default=0.5)
    timeliness_score: float = Field(ge=0.0, le=1.0, default=0.5)
    user_engagement_score: float = Field(ge=0.0, le=1.0, default=0.0)

    # Security
    security_classification: SecurityClassification = SecurityClassification.INTERNAL

    # Metadata
    trigger_reason: str
    source_system: str = "pinakontext_nudge_engine"

    def is_expired(self) -> bool:
        """Check if nudge has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def is_delivered(self) -> bool:
        """Check if nudge has been delivered."""
        return self.delivered_at is not None

    def is_acknowledged(self) -> bool:
        """Check if nudge has been acknowledged."""
        return self.acknowledged_at is not None

    def calculate_effectiveness_score(self) -> float:
        """Calculate overall effectiveness score."""
        if not self.is_delivered():
            return 0.0

        # Weight different factors
        weights = {"relevance": 0.4, "timeliness": 0.3, "engagement": 0.3}

        return (
            weights["relevance"] * self.relevance_score
            + weights["timeliness"] * self.timeliness_score
            + weights["engagement"] * self.user_engagement_score
        )


class NudgeDeliveryResult(BaseModel):
    """Result of nudge delivery attempt."""

    nudge_id: str
    delivery_method: str
    success: bool
    delivered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Delivery details
    channel: str  # e.g., "cli", "api", "notification"
    recipient: str
    error_message: Optional[str] = None

    # User response (if any)
    user_response: Optional[str] = None
    response_time_seconds: Optional[int] = None

    # Quality metrics
    delivery_confidence: float = Field(ge=0.0, le=1.0, default=1.0)


class INudgeStore(Protocol):
    """Protocol for nudge storage implementations."""

    async def store_nudge(self, nudge: Nudge) -> bool:
        """Store a nudge."""
        ...

    async def get_nudge(self, nudge_id: str) -> Optional[Nudge]:
        """Retrieve a nudge by ID."""
        ...

    async def get_pending_nudges(self, user_id: str, limit: int = 10) -> List[Nudge]:
        """Get pending nudges for a user."""
        ...

    async def update_nudge_status(self, nudge_id: str, status: str) -> bool:
        """Update nudge delivery/acknowledgment status."""
        ...

    async def get_nudge_templates(self, active_only: bool = True) -> List[NudgeTemplate]:
        """Get available nudge templates."""
        ...

    async def store_delivery_result(self, result: NudgeDeliveryResult) -> bool:
        """Store delivery result."""
        ...


class INudgeDelivery(Protocol):
    """Protocol for nudge delivery mechanisms."""

    async def deliver_nudge(self, nudge: Nudge) -> NudgeDeliveryResult:
        """Deliver a nudge through this channel."""
        ...

    def get_delivery_channel(self) -> str:
        """Get the delivery channel name."""
        ...

    def is_available(self) -> bool:
        """Check if delivery channel is available."""
        ...

from __future__ import annotations

from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class VectorClock(BaseModel):
    agent: str
    counter: int


class BaseMemory(BaseModel):
    id: str
    layer: Literal[
        "working",
        "session",
        "event",
        "changelog",
        "episodic",
        "semantic",
        "procedural",
        "rag",
    ]
    tenant: Optional[str] = None
    agent: Optional[str] = None
    scope: Optional[Literal["user", "project", "thread", "org"]] = None
    timestamp: Optional[str] = None
    ttl_seconds: Optional[int] = 0
    tags: List[str] = []
    pii_flags: List[str] = []
    metadata: Dict[str, Any] = {}
    content: Optional[str] = None
    embedding: Optional[List[float]] = None
    sparse: Optional[Dict[str, Any]] = None
    hash: Optional[str] = None
    prev_hash: Optional[str] = None
    op_id: Optional[str] = None
    vector_clock: Optional[VectorClock] = None
    index_version: Optional[str] = Field(default="sem-v1")
    redaction_policy: Optional[str] = None


class WorkingMemory(BaseMemory):
    layer: Literal["working"] = "working"
    relevance_score: Optional[float] = 0.0


class SessionMemory(BaseMemory):
    layer: Literal["session"] = "session"
    session_id: Optional[str] = None
    parent_id: Optional[str] = None
    summary: Optional[str] = None


class EventMemory(BaseMemory):
    layer: Literal["event"] = "event"
    event_type: Literal["tool_call", "decision", "error"]
    outcome: Optional[str] = None
    payload: Dict[str, Any] = {}
    latency_ms: Optional[int] = 0
    cost_tokens: Optional[int] = 0


class ChangelogMemory(BaseMemory):
    layer: Literal["changelog"] = "changelog"
    change_type: Literal["create", "update", "delete", "redact"]
    actor: Optional[str] = None
    target_id: Optional[str] = None
    reason: Optional[str] = None


class EpisodicMemory(BaseModel):
    layer: Literal["episodic"] = "episodic"
    salience: Optional[int] = 0


class SemanticMemory(BaseMemory):
    layer: Literal["semantic"] = "semantic"
    source: Optional[Literal["manual", "ingested", "learned"]] = "manual"
    confidence: Optional[float] = 0.0
    attribution: List[str] = []


class ProceduralMemory(BaseMemory):
    layer: Literal["procedural"] = "procedural"
    skill_id: Optional[str] = None
    steps: List[str] = []
    parameters: Dict[str, Any] = {}
    success_rate: Optional[float] = 0.0


class RagMemory(BaseMemory):
    layer: Literal["rag"] = "rag"
    query: Optional[str] = None
    external_source: Optional[str] = None
    scores: List[float] = []
    retrieved_docs: List[Dict[str, Any]] = []


def json_schema_for_memory() -> Dict[str, Any]:
    """Return a composite JSON Schema by composing layer models."""
    models = [
        WorkingMemory,
        SessionMemory,
        EventMemory,
        ChangelogMemory,
        EpisodicMemory,
        SemanticMemory,
        ProceduralMemory,
        RagMemory,
    ]
    defs: Dict[str, Any] = {m.__name__: m.model_json_schema() for m in models}
    union_schema = {
        "oneOf": [{"$ref": f"#/definitions/{m.__name__}"} for m in models],
        "definitions": defs,
    }
    return union_schema

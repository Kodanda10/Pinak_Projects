from typing import Optional, List, Any
from sqlalchemy import Column, String, Integer, Text, JSON, Float, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import AsyncAttrs

class Base(AsyncAttrs, DeclarativeBase):
    pass

class SemanticMemory(Base):
    __tablename__ = "memories_semantic"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    tags: Mapped[Optional[str]] = mapped_column(Text) # JSON list
    embedding_id: Mapped[Optional[int]] = mapped_column(Integer)
    agent_id: Mapped[Optional[str]] = mapped_column(String)
    client_id: Mapped[Optional[str]] = mapped_column(String)
    client_name: Mapped[Optional[str]] = mapped_column(String)
    tenant: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)

class EpisodicMemory(Base):
    __tablename__ = "memories_episodic"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    goal: Mapped[Optional[str]] = mapped_column(Text)
    outcome: Mapped[Optional[str]] = mapped_column(Text)
    plan: Mapped[Optional[str]] = mapped_column(Text) # JSON
    steps: Mapped[Optional[str]] = mapped_column(Text) # JSON list
    salience: Mapped[Optional[int]] = mapped_column(Integer)
    embedding_id: Mapped[Optional[int]] = mapped_column(Integer)
    agent_id: Mapped[Optional[str]] = mapped_column(String)
    client_id: Mapped[Optional[str]] = mapped_column(String)
    client_name: Mapped[Optional[str]] = mapped_column(String)
    tenant: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)

class ProceduralMemory(Base):
    __tablename__ = "memories_procedural"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    skill_name: Mapped[str] = mapped_column(Text, nullable=False)
    trigger: Mapped[Optional[str]] = mapped_column(Text)
    steps: Mapped[Optional[str]] = mapped_column(Text) # JSON
    description: Mapped[Optional[str]] = mapped_column(Text)
    code_snippet: Mapped[Optional[str]] = mapped_column(Text)
    embedding_id: Mapped[Optional[int]] = mapped_column(Integer)
    agent_id: Mapped[Optional[str]] = mapped_column(String)
    client_id: Mapped[Optional[str]] = mapped_column(String)
    client_name: Mapped[Optional[str]] = mapped_column(String)
    tenant: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)

class RAGMemory(Base):
    __tablename__ = "memories_rag"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    query: Mapped[Optional[str]] = mapped_column(Text)
    external_source: Mapped[Optional[str]] = mapped_column(Text)
    content: Mapped[Optional[str]] = mapped_column(Text)
    agent_id: Mapped[Optional[str]] = mapped_column(String)
    client_id: Mapped[Optional[str]] = mapped_column(String)
    client_name: Mapped[Optional[str]] = mapped_column(String)
    tenant: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)

class WorkingMemory(Base):
    __tablename__ = "working_memory"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    key: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[str] = mapped_column(Text, nullable=False) # JSON
    agent_id: Mapped[Optional[str]] = mapped_column(String)
    client_id: Mapped[Optional[str]] = mapped_column(String)
    client_name: Mapped[Optional[str]] = mapped_column(String)
    tenant: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)
    expires_at: Mapped[Optional[str]] = mapped_column(String)

class EventLog(Base):
    __tablename__ = "logs_events"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[Optional[str]] = mapped_column(Text) # JSON
    tenant: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    ts: Mapped[str] = mapped_column(String, nullable=False)

class SessionLog(Base):
    __tablename__ = "logs_session"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    tool_calls: Mapped[Optional[str]] = mapped_column(Text) # JSON
    agent_id: Mapped[Optional[str]] = mapped_column(String)
    client_id: Mapped[Optional[str]] = mapped_column(String)
    client_name: Mapped[Optional[str]] = mapped_column(String)
    parent_client_id: Mapped[Optional[str]] = mapped_column(String)
    child_client_id: Mapped[Optional[str]] = mapped_column(String)
    tenant: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    ts: Mapped[str] = mapped_column(String, nullable=False)
    expires_at: Mapped[Optional[str]] = mapped_column(String)

class ClientRegistry(Base):
    __tablename__ = "clients_registry"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    client_id: Mapped[str] = mapped_column(String, nullable=False)
    client_name: Mapped[Optional[str]] = mapped_column(String)
    parent_client_id: Mapped[Optional[str]] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, nullable=False)
    metadata_: Mapped[Optional[str]] = mapped_column("metadata", Text) # JSON
    tenant: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)
    last_seen: Mapped[Optional[str]] = mapped_column(String)

class AgentLog(Base):
    __tablename__ = "logs_agents"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[str] = mapped_column(String, nullable=False)
    client_name: Mapped[str] = mapped_column(String, nullable=False)
    client_id: Mapped[Optional[str]] = mapped_column(String)
    parent_client_id: Mapped[Optional[str]] = mapped_column(String)
    hostname: Mapped[Optional[str]] = mapped_column(String)
    pid: Mapped[Optional[str]] = mapped_column(String)
    status: Mapped[str] = mapped_column(String, nullable=False)
    meta: Mapped[Optional[str]] = mapped_column(Text) # JSON
    tenant: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    last_seen: Mapped[str] = mapped_column(String, nullable=False)

class AccessLog(Base):
    __tablename__ = "logs_access"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    agent_id: Mapped[Optional[str]] = mapped_column(String)
    client_name: Mapped[Optional[str]] = mapped_column(String)
    client_id: Mapped[Optional[str]] = mapped_column(String)
    parent_client_id: Mapped[Optional[str]] = mapped_column(String)
    child_client_id: Mapped[Optional[str]] = mapped_column(String)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    target_layer: Mapped[Optional[str]] = mapped_column(String)
    query: Mapped[Optional[str]] = mapped_column(Text)
    memory_id: Mapped[Optional[str]] = mapped_column(String)
    result_count: Mapped[Optional[int]] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String, nullable=False)
    detail: Mapped[Optional[str]] = mapped_column(Text)
    tenant: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    ts: Mapped[str] = mapped_column(String, nullable=False)

class Quarantine(Base):
    __tablename__ = "memory_quarantine"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    layer: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[str] = mapped_column(Text, nullable=False) # JSON
    status: Mapped[str] = mapped_column(String, nullable=False)
    agent_id: Mapped[Optional[str]] = mapped_column(String)
    client_id: Mapped[Optional[str]] = mapped_column(String)
    client_name: Mapped[Optional[str]] = mapped_column(String)
    validation_errors: Mapped[Optional[str]] = mapped_column(Text) # JSON
    tenant: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    reviewed_at: Mapped[Optional[str]] = mapped_column(String)
    reviewed_by: Mapped[Optional[str]] = mapped_column(String)

class AuditLog(Base):
    __tablename__ = "logs_audit"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[str] = mapped_column(Text, nullable=False) # JSON
    prev_hash: Mapped[Optional[str]] = mapped_column(String)
    hash: Mapped[str] = mapped_column(String, nullable=False)
    ts: Mapped[str] = mapped_column(String, nullable=False)

class ClientIssue(Base):
    __tablename__ = "logs_client_issues"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    client_id: Mapped[str] = mapped_column(String, nullable=False)
    client_name: Mapped[Optional[str]] = mapped_column(String)
    agent_id: Mapped[Optional[str]] = mapped_column(String)
    parent_client_id: Mapped[Optional[str]] = mapped_column(String)
    child_client_id: Mapped[Optional[str]] = mapped_column(String)
    layer: Mapped[Optional[str]] = mapped_column(String)
    error_code: Mapped[str] = mapped_column(String, nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    payload: Mapped[Optional[str]] = mapped_column(Text)
    metadata_: Mapped[Optional[str]] = mapped_column("metadata", Text)
    status: Mapped[str] = mapped_column(String, nullable=False)
    tenant: Mapped[str] = mapped_column(String, nullable=False)
    project_id: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    resolved_at: Mapped[Optional[str]] = mapped_column(String)
    resolved_by: Mapped[Optional[str]] = mapped_column(String)
    resolution: Mapped[Optional[str]] = mapped_column(Text)

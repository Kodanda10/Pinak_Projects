from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union

class MemoryCreate(BaseModel):
    content: str
    tags: Optional[List[str]] = None

class MemoryRead(BaseModel):
    id: str
    content: str
    tags: List[str] = []
    tenant: str
    project_id: str
    created_at: str

    model_config = {
        "from_attributes": True
    }

class MemorySearchResult(MemoryRead):
    distance: float
    metadata: Optional[Dict[str, Any]] = None

class EpisodicCreate(BaseModel):
    content: str
    salience: int = 0
    goal: Optional[str] = None
    plan: Optional[List[str]] = None
    outcome: Optional[str] = None
    tool_logs: Optional[List[Dict[str, Any]]] = None

class ProceduralCreate(BaseModel):
    skill_name: str
    steps: List[str]
    trigger: Optional[str] = None
    code_snippet: Optional[str] = None

class RAGCreate(BaseModel):
    query: str
    external_source: str
    content: str

class EventCreate(BaseModel):
    event_type: str
    payload: Dict[str, Any]

class SessionCreate(BaseModel):
    session_id: str
    content: str
    role: str = "user"

class WorkingCreate(BaseModel):
    content: str

class ContextResponse(BaseModel):
    semantic: List[Dict[str, Any]]
    episodic: List[Dict[str, Any]]
    procedural: List[Dict[str, Any]]
    working: List[Dict[str, Any]]

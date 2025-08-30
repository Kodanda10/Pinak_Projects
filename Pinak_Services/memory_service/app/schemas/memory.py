import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class MemoryCreate(BaseModel):
    content: str
    tags: Optional[List[str]] = []
    metadata: Optional[Dict[str, Any]] = {}


class MemoryOut(BaseModel):
    id: str
    faiss_id: Optional[int]
    content: str
    tags: List[str]
    created_at: datetime.datetime
    redacted: Optional[str]
    metadata: Optional[Dict[str, Any]] = {}
    distance: Optional[float] = None

    class Config:
        from_attributes = True  # For Pydantic V2


class MemorySearchResult(MemoryOut):
    distance: float

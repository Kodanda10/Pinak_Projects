from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import datetime

class MemoryCreate(BaseModel):
    content: str
    tags: Optional[List[str]] = []
    metadata: Optional[Dict[str, Any]] = {}

class MemoryRead(BaseModel):
    id: str
    content: str
    tags: List[str]

    class Config:
        from_attributes = True

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
        from_attributes = True

class MemorySearchResult(MemoryOut):
    pass
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import datetime

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

    class Config:
        from_attributes = True # For Pydantic V2

class MemorySearchResult(MemoryOut):
    distance: float
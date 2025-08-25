from pydantic import BaseModel
from typing import List, Optional

class MemoryCreate(BaseModel):
    content: str
    tags: Optional[List[str]] = []

class MemoryRead(BaseModel):
    id: str
    content: str
    tags: List[str]

    class Config:
        from_attributes = True

class MemorySearchResult(MemoryRead):
    distance: float
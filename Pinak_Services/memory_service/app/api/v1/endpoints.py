from fastapi import APIRouter, status
from app.core.schemas import MemoryCreate, MemoryRead, MemorySearchResult
from app.services.memory_service import memory_service
from typing import List

router = APIRouter()

@router.post("/add", response_model=MemoryRead, status_code=status.HTTP_201_CREATED)
def add_memory(memory: MemoryCreate):
    """API endpoint to add a new memory."""
    return memory_service.add_memory(memory)

@router.get("/search", response_model=List[MemorySearchResult])
def search_memory(query: str, k: int = 5):
    """API endpoint to search for relevant memories."""
    return memory_service.search_memory(query=query, k=k)
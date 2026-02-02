import pytest
from pydantic import ValidationError
from app.core.schemas import MemoryCreate, EpisodicCreate, RAGCreate

def test_memory_create_max_length():
    # Valid
    MemoryCreate(content="a"*1000)

    # Invalid
    with pytest.raises(ValidationError):
        MemoryCreate(content="a"*100001)

def test_episodic_create_limits():
    # Valid salience
    EpisodicCreate(content="a", salience=10)

    # Invalid salience
    with pytest.raises(ValidationError):
        EpisodicCreate(content="a", salience=11)

    with pytest.raises(ValidationError):
        EpisodicCreate(content="a", salience=-1)

def test_rag_create_limits():
    # Valid
    RAGCreate(query="q", external_source="s", content="c")

    # Invalid query
    with pytest.raises(ValidationError):
        RAGCreate(query="a"*4097, external_source="s", content="c")

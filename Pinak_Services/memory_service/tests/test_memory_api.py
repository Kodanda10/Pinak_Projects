import pytest
from httpx import AsyncClient, ASGITransport
from asgi_lifespan import LifespanManager
import json
import os
import tempfile
from unittest.mock import patch

from app.main import app
from app.services.memory_service import MemoryService

@pytest.mark.asyncio
async def test_add_memory():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/memory/add", json={"content": "test memory", "tags": ["test"]})

    assert response.status_code == 201
    assert "id" in response.json()
    assert response.json()["content"] == "test memory"

@pytest.mark.asyncio
async def test_retrieve_memory():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            memory_content = "The sky is blue on a clear day."
            add_response = await ac.post("/api/v1/memory/add", json={"content": memory_content})
            assert add_response.status_code == 201

            search_response = await ac.get("/api/v1/memory/search?query=What color is the sky?")

    assert search_response.status_code == 200
    response_data = search_response.json()
    assert isinstance(response_data, list)
    assert len(response_data) > 0
    assert response_data[0]['content'] == memory_content

@pytest.mark.asyncio
async def test_add_episodic_memory():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/memory/episodic/add", json={"content": "episodic test", "salience": 5})

    assert response.status_code == 201
    data = response.json()
    assert "content" in data
    assert data["salience"] == 5

@pytest.mark.asyncio
async def test_list_episodic_memory():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # Add first
            await ac.post("/api/v1/memory/episodic/add", json={"content": "episodic 1", "salience": 3})
            await ac.post("/api/v1/memory/episodic/add", json={"content": "episodic 2", "salience": 7})

            response = await ac.get("/api/v1/memory/episodic/list")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2

@pytest.mark.asyncio
async def test_add_procedural_memory():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/memory/procedural/add", 
                                   json={"skill_id": "debug", "steps": ["step1", "step2"]})

    assert response.status_code == 201
    data = response.json()
    assert data["skill_id"] == "debug"
    assert data["steps"] == ["step1", "step2"]

@pytest.mark.asyncio
async def test_add_rag_memory():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/memory/rag/add", 
                                   json={"query": "how to test", "external_source": "docs"})

    assert response.status_code == 201
    data = response.json()
    assert data["query"] == "how to test"
    assert data["external_source"] == "docs"

@pytest.mark.asyncio
async def test_search_v2_multiple_layers():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # Add data to different layers
            await ac.post("/api/v1/memory/episodic/add", json={"content": "test episodic", "salience": 5})
            await ac.post("/api/v1/memory/procedural/add", json={"skill_id": "test skill", "steps": ["a", "b"]})
            await ac.post("/api/v1/memory/rag/add", json={"query": "test query", "external_source": "kb"})

            response = await ac.get("/api/v1/memory/search_v2?query=test&layers=episodic,procedural,rag")

    assert response.status_code == 200
    data = response.json()
    assert "episodic" in data
    assert "procedural" in data
    assert "rag" in data

@pytest.mark.asyncio
async def test_add_event():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/memory/event", 
                                   json={"type": "user_action", "action": "login"})

    assert response.status_code == 201
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_session_memory():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # Add session memory
            response = await ac.post("/api/v1/memory/session/add", 
                                   json={"session_id": "test_session", "content": "session content"})
            assert response.status_code == 201

            # List session memory
            response = await ac.get("/api/v1/memory/session/list?session_id=test_session")
            assert response.status_code == 200
            data = response.json()
            assert len(data) > 0
            assert data[0]["content"] == "session content"

@pytest.mark.asyncio
async def test_working_memory():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/memory/working/add", 
                                   json={"content": "working content", "ttl_seconds": 60})
            assert response.status_code == 201

            response = await ac.get("/api/v1/memory/working/list")
            assert response.status_code == 200
            data = response.json()
            assert len(data) > 0

def test_memory_service_initialization():
    """Test MemoryService initialization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump({
                "embedding_model": "all-MiniLM-L6-v2",
                "vector_db_path": os.path.join(temp_dir, "memory.faiss"),
                "metadata_db_path": os.path.join(temp_dir, "metadata.json"),
                "redis_host": "localhost"
            }, f)
        
        service = MemoryService(config_path)
        assert service.config is not None
        assert service.model is not None

def test_memory_service_add_and_search():
    """Test adding and searching memory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump({
                "embedding_model": "all-MiniLM-L6-v2",
                "vector_db_path": os.path.join(temp_dir, "memory.faiss"),
                "metadata_db_path": os.path.join(temp_dir, "metadata.json"),
                "redis_host": "localhost"
            }, f)
        
        service = MemoryService(config_path)
        
        # Mock Redis to avoid connection
        with patch.object(service, 'redis_client', None):
            from app.core.schemas import MemoryCreate
            memory = MemoryCreate(content="test content", tags=["test"])
            result = service.add_memory(memory)
            assert result.content == "test content"
            
            results = service.search_memory("test", k=1)
            assert len(results) > 0

@pytest.mark.asyncio
async def test_events_listing():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # Add events
            await ac.post("/api/v1/memory/event", json={"type": "test_event", "data": "value1"})
            await ac.post("/api/v1/memory/event", json={"type": "test_event", "data": "value2"})

            response = await ac.get("/api/v1/memory/events")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) >= 2

# Integration test for full workflow
@pytest.mark.asyncio
async def test_full_memory_workflow():
    """Test complete memory workflow across layers."""
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # 1. Add semantic memory
            await ac.post("/api/v1/memory/add", json={"content": "Python is great", "tags": ["python"]})
            
            # 2. Add episodic memory
            await ac.post("/api/v1/memory/episodic/add", json={"content": "Learned Python today", "salience": 9})
            
            # 3. Add procedural memory
            await ac.post("/api/v1/memory/procedural/add", json={"skill_id": "python_debug", "steps": ["print", "debug"]})
            
            # 4. Add event
            await ac.post("/api/v1/memory/event", json={"type": "learning", "topic": "python"})
            
            # 5. Search across layers
            response = await ac.get("/api/v1/memory/search_v2?query=python&layers=episodic,procedural")
            assert response.status_code == 200
            data = response.json()
            assert len(data.get("episodic", [])) > 0
            assert len(data.get("procedural", [])) > 0

@pytest.mark.asyncio
async def test_add_episodic_memory():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/memory/episodic/add", json={"content": "episodic test", "salience": 5})
    
    assert response.status_code == 201
    data = response.json()
    assert "content" in data
    assert data["content"] == "episodic test"

@pytest.mark.asyncio
async def test_list_episodic_memory():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # Add first
            await ac.post("/api/v1/memory/episodic/add", json={"content": "episodic test", "salience": 5})
            # List
            response = await ac.get("/api/v1/memory/episodic/list")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

@pytest.mark.asyncio
async def test_add_procedural_memory():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/memory/procedural/add", json={"skill_id": "build", "steps": ["step1", "step2"]})
    
    assert response.status_code == 201
    data = response.json()
    assert "skill_id" in data
    assert data["skill_id"] == "build"

@pytest.mark.asyncio
async def test_add_rag_memory():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/memory/rag/add", json={"query": "faq", "external_source": "kb"})
    
    assert response.status_code == 201
    data = response.json()
    assert "query" in data
    assert data["query"] == "faq"

@pytest.mark.asyncio
async def test_search_v2():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # Add some data
            await ac.post("/api/v1/memory/episodic/add", json={"content": "test episodic", "salience": 5})
            await ac.post("/api/v1/memory/procedural/add", json={"skill_id": "test", "steps": ["a"]})
            # Search
            response = await ac.get("/api/v1/memory/search_v2?query=test&layers=episodic,procedural")
    
    assert response.status_code == 200
    data = response.json()
    assert "episodic" in data
    assert "procedural" in data

@pytest.mark.asyncio
async def test_add_event():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/memory/event", json={"type": "test", "data": "value"})
    
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "ok"

@pytest.mark.asyncio
async def test_add_session():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/memory/session/add", json={"session_id": "test", "content": "session content"})
    
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "ok"

@pytest.mark.asyncio
async def test_add_working():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/memory/working/add", json={"content": "working content"})
    
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "ok"
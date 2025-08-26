import os, pytest
os.environ.setdefault('USE_MOCK_EMBEDDINGS','true')
from httpx import AsyncClient, ASGITransport
from asgi_lifespan import LifespanManager

from app.main import app 

@pytest.mark.asyncio
async def test_add_memory():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/api/v1/memory/add", json={"content": "test memory", "tags": ["test"]})
    
    assert response.status_code == 201 # 201 Created
    assert "id" in response.json()
    assert response.json()["content"] == "test memory"

@pytest.mark.asyncio
async def test_retrieve_memory():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # 1. Add a known memory first
            memory_content = "The sky is blue on a clear day."
            add_response = await ac.post("/api/v1/memory/add", json={"content": memory_content})
            assert add_response.status_code == 201

            # 2. Try to retrieve it
            search_response = await ac.get(f"/api/v1/memory/search?query=What color is the sky?")
            
    assert search_response.status_code == 200
    response_data = search_response.json()
    assert isinstance(response_data, list)
    assert len(response_data) > 0
    assert response_data[0]['content'] == memory_content

import os

import pytest

os.environ.setdefault("USE_MOCK_EMBEDDINGS", "true")
import pytest_asyncio
from app.main import app
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


@pytest.mark.asyncio
async def test_add_memory(client_and_memory_service):
    """Test adding a memory via API."""
    client, service = client_and_memory_service

    response = await client.post(
        "/api/v1/memory/add", json={"content": "test memory", "tags": ["test"]}
    )

    assert response.status_code == 201  # 201 Created
    response_data = response.json()
    assert "id" in response_data
    assert response_data["content"] == "test memory"


@pytest.mark.asyncio
async def test_retrieve_memory(client_and_memory_service):
    """Test retrieving memories via API search."""
    client, service = client_and_memory_service

    # 1. Add a known memory first
    memory_content = "The sky is blue on a clear day."
    add_response = await client.post(
        "/api/v1/memory/add", json={"content": memory_content}
    )
    assert add_response.status_code == 201

    # 2. Try to retrieve it
    search_response = await client.get("/api/v1/memory/search?query=sky is blue")

    assert search_response.status_code == 200
    response_data = search_response.json()
    assert isinstance(response_data, list)
    assert len(response_data) > 0
    assert any(item["content"] == memory_content for item in response_data)

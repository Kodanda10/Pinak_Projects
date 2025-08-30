import asyncio
import json
import math
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from app.core.schemas import MemoryCreate, MemorySearchResult
from app.db.models import Base
from app.main import app as fastapi_app
from app.services.memory_service import MemoryService
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio


async def _ensure_ready(service) -> None:
    """Optional: quickly sanity-check the service before each test."""
    return None


# -----------------------------
# Tests
# -----------------------------


async def test_faiss_add_and_search_basic(setup_teardown_db):
    service = setup_teardown_db

    # Arrange
    memories_to_add = [
        {"content": "The quick brown fox jumps over the lazy dog.", "tags": ["animal"]},
        {
            "content": "Artificial intelligence is transforming industries.",
            "tags": ["tech"],
        },
        {
            "content": "Quantum computing promises revolutionary changes.",
            "tags": ["tech"],
        },
        {"content": "A cat sat on the mat.", "tags": ["animal"]},
    ]

    session = service.Session()  # Get a session from the service
    try:
        for memory_data in memories_to_add:
            service.add_memory(
                MemoryCreate(**memory_data), session
            )  # Pass session as second parameter
        session.commit()  # Commit the changes after all additions
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

    # Act
    session = service.Session()  # Get a new session for search
    try:
        hits = service.search_memory(query="animals on a farm", k=2, db=session)
    finally:
        session.close()

    # Assert
    assert len(hits) > 0
    found_relevant = False
    for result in hits:
        if (
            "fox" in result.content.lower()
            or "dog" in result.content.lower()
            or "cat" in result.content.lower()
        ):
            found_relevant = True
            break
    assert (
        found_relevant
    ), f"No relevant animal memory found for query 'animals on a farm'"


async def test_api_search_endpoint(client_and_memory_service):
    client, service = client_and_memory_service
    memories_to_add = [
        {"content": "The quick brown fox jumps over the lazy dog.", "tags": ["animal"]},
        {
            "content": "Artificial intelligence is transforming industries.",
            "tags": ["tech"],
        },
        {
            "content": "Quantum computing promises revolutionary changes.",
            "tags": ["tech"],
        },
        {"content": "A cat sat on the mat.", "tags": ["animal"]},
    ]
    for memory_data in memories_to_add:
        response = await client.post("/api/v1/memory/add", json=memory_data)
        assert response.status_code == 201

    search_query = "animals on a farm"
    search_response = await client.get(
        f"/api/v1/memory/search?query={search_query}&k=2"
    )
    assert search_response.status_code == 200
    results = search_response.json()
    assert isinstance(results, list)
    assert len(results) > 0

    found_relevant = False
    for result in results:
        if (
            "fox" in result["content"].lower()
            or "dog" in result["content"].lower()
            or "cat" in result["content"].lower()
        ):
            found_relevant = True
            break
    assert found_relevant, f"No relevant animal memory found for query '{search_query}'"

    search_query_tech = "future of technology"
    search_response_tech = await client.get(
        f"/api/v1/memory/search?query={search_query_tech}&k=2"
    )
    assert search_response_tech.status_code == 200
    results_tech = search_response_tech.json()
    assert isinstance(results_tech, list)
    assert len(results_tech) > 0
    found_relevant_tech = False
    for result in results_tech:
        if (
            "intelligence" in result["content"].lower()
            or "quantum" in result["content"].lower()
        ):
            found_relevant_tech = True
            break
    assert (
        found_relevant_tech
    ), f"No relevant tech memory found for query '{search_query_tech}''"


async def test_redis_caching(client_and_memory_service):
    ac, ms = client_and_memory_service

    # Create a mock Redis client
    from unittest.mock import MagicMock

    mock_redis_client = MagicMock()
    mock_redis_client.ping.return_value = True
    mock_redis_client.get.return_value = None  # Start with cache miss

    # Set the Redis client on the service
    ms.redis_client = mock_redis_client

    # Test Cache Miss and Set
    memories_to_add = [
        {"content": "Cached memory 1.", "tags": ["cache"]},
        {"content": "Cached memory 2.", "tags": ["cache"]},
    ]
    for memory_data in memories_to_add:
        response = await ac.post("/api/v1/memory/add", json=memory_data)
        assert response.status_code == 201

    search_query = "Cached memory"
    search_response = await ac.get(f"/api/v1/memory/search?query={search_query}&k=2")
    assert search_response.status_code == 200
    results = search_response.json()
    print(f"Search results: {results}")  # Debug print
    assert len(results) > 0

    mock_redis_client.get.assert_called_with(f"search:{search_query}:{2}")
    print(f"Redis get called: {mock_redis_client.get.called}")  # Debug print
    print(f"Redis setex called: {mock_redis_client.setex.called}")  # Debug print
    assert mock_redis_client.setex.called
    call_args, call_kwargs = mock_redis_client.setex.call_args
    assert call_args[0] == f"search:{search_query}:{2}"
    set_value = json.loads(call_args[2])
    # Since results is already a list of dicts from the API response, compare directly
    assert set_value == results

    # Test Cache Hit
    mock_redis_client.get.return_value = json.dumps(results)
    mock_redis_client.setex.reset_mock()  # Reset setex mock for this part

    search_response_cached = await ac.get(
        f"/api/v1/memory/search?query={search_query}&k=2"
    )
    assert search_response_cached.status_code == 200
    results_cached = search_response_cached.json()
    assert results_cached == results

    mock_redis_client.get.assert_called_with(f"search:{search_query}:{2}")
    mock_redis_client.setex.assert_not_called()

    # Test Cache Invalidation on add_memory
    mock_redis_client.scan_iter.return_value = [b"search:cached data:2"]
    mock_redis_client.delete.reset_mock()  # Reset delete mock

    response_add_new = await ac.post(
        "/api/v1/memory/add",
        json={"content": "New memory to invalidate cache.", "tags": ["new"]},
    )
    assert response_add_new.status_code == 201

    mock_redis_client.scan_iter.assert_called_with("search:*")
    mock_redis_client.delete.assert_called_with(b"search:cached data:2")

    # Test Cache Invalidation on redact_memory (placeholder)
    pass

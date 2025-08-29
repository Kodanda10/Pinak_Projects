import pytest_asyncio
import asyncio
import math
import pytest
from httpx import AsyncClient
from unittest.mock import MagicMock, patch

from app.main import app as fastapi_app
from app.services.memory_service import MemoryService, Base
from app.core.schemas import MemoryCreate, MemorySearchResult

pytestmark = pytest.mark.asyncio

async def _ensure_ready(service) -> None:
    """Optional: quickly sanity-check the service before each test."""
    return None

@pytest_asyncio.fixture
async def setup_teardown_db(tmp_path, fresh_memory_service_factory):
    """
    Creates an isolated on-disk SQLite DB for each test, then drops & recreates all tables.
    Yields a ready-to-use MemoryService instance.
    """
    db_path = tmp_path / "test.db"
    service = await fresh_memory_service_factory(db_url=f"sqlite:///{db_path}")
    await service.drop_all()
    await service.create_all()

    await _ensure_ready(service)
    try:
        yield service
    finally:
        try:
            await service.drop_all()
        finally:
            await service.aclose()

@pytest_asyncio.fixture
async def client_and_memory_service(setup_teardown_db, fresh_test_client_factory):
    """
    Provides (httpx.AsyncClient, MemoryService) wired together so API uses THIS service instance.
    """
    service = setup_teardown_db
    client = await fresh_test_client_factory(service)
    try:
        yield client, service
    finally:
        await client.aclose()

# -----------------------------
# Tests
# -----------------------------

async def test_faiss_add_and_search_basic(setup_teardown_db):
    service = setup_teardown_db

    # Arrange
    memories_to_add = [
        {"content": "The quick brown fox jumps over the lazy dog.", "tags": ["animal"]},
        {"content": "Artificial intelligence is transforming industries.", "tags": ["tech"]},
        {"content": "Quantum computing promises revolutionary changes.", "tags": ["tech"]},
        {"content": "A cat sat on the mat.", "tags": ["animal"]},
    ]
    
    session = service.Session() # Get a session from the service
    try:
        for memory_data in memories_to_add:
            service.add_memory(session, MemoryCreate(**memory_data)) # Pass the session
        session.commit() # Commit the changes after all additions
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

    # Act
    hits = await service.search_memory(query="animals on a farm", k=2)

    # Assert
    assert len(hits) > 0
    found_relevant = False
    for result in hits:
        if "fox" in result['content'].lower() or "dog" in result['content'].lower() or "cat" in result['content'].lower():
            found_relevant = True
            break
    assert found_relevant, f"No relevant animal memory found for query 'animals on a farm'"


async def test_api_search_endpoint(client_and_memory_service):
    client, service = client_and_memory_service
    memories_to_add = [
        {"content": "The quick brown fox jumps over the lazy dog.", "tags": ["animal"]},
        {"content": "Artificial intelligence is transforming industries.", "tags": ["tech"]},
        {"content": "Quantum computing promises revolutionary changes.", "tags": ["tech"]},
        {"content": "A cat sat on the mat.", "tags": ["animal"]},
    ]
    for memory_data in memories_to_add:
        response = await client.post("/api/v1/memory/add", json=memory_data)
        assert response.status_code == 201

    search_query = "animals on a farm"
    search_response = await client.get(f"/api/v1/memory/search?query={search_query}&k=2")
    assert search_response.status_code == 200
    results = search_response.json()
    assert isinstance(results, list)
    assert len(results) > 0

    found_relevant = False
    for result in results:
        if "fox" in result['content'].lower() or "dog" in result['content'].lower() or "cat" in result['content'].lower():
            found_relevant = True
            break
    assert found_relevant, f"No relevant animal memory found for query '{search_query}'"

    search_query_tech = "future of technology"
    search_response_tech = await client.get(f"/api/v1/memory/search?query={search_query_tech}&k=2")
    assert search_response_tech.status_code == 200
    results_tech = search_response_tech.json()
    assert isinstance(results_tech, list)
    assert len(results_tech) > 0
    found_relevant_tech = False
    for result in results_tech:
        if "intelligence" in result['content'].lower() or "quantum" in result['content'].lower():
            found_relevant_tech = True
            break
    assert found_relevant_tech, f"No relevant tech memory found for query '{search_query_tech}''"

async def test_redis_caching(client_and_memory_service):
    ac, ms = client_and_memory_service

    with patch.object(ms, 'redis_client') as mock_redis_client:
        mock_redis_client.return_value = MagicMock()
        mock_redis_client.ping.return_value = True

        # Test Cache Miss and Set
        mock_redis_client.get.return_value = None
        
        memories_to_add = [
            {"content": "Cached memory 1.", "tags": ["cache"]},
            {"content": "Cached memory 2.", "tags": ["cache"]},
        ]
        for memory_data in memories_to_add:
            response = await ac.post("/api/v1/memory/add", json=memory_data)
            assert response.status_code == 201

        search_query = "cached data"
        search_response = await ac.get(f"/api/v1/memory/search?query={search_query}&k=2")
        assert search_response.status_code == 200
        results = search_response.json()
        assert len(results) > 0

        mock_redis_client.get.assert_called_with(f"search:{search_query}:{2}")
        assert mock_redis_client.setex.called
        call_args, call_kwargs = mock_redis_client.setex.call_args
        assert call_args[0] == f"search:{search_query}:{2}"
        set_value = json.loads(call_args[2])
        expected_set_value = [json.loads(r.model_dump_json()) for r in results]
        assert set_value == expected_set_value

        # Test Cache Hit
        mock_redis_client.get.return_value = json.dumps([json.dumps(r) for r in results])
        mock_redis_client.setex.reset_mock() # Reset setex mock for this part

        search_response_cached = await ac.get(f"/api/v1/memory/search?query={search_query}&k=2")
        assert search_response_cached.status_code == 200
        results_cached = search_response_cached.json()
        assert results_cached == results

        mock_redis_client.get.assert_called_with(f"search:{search_query}:{2}")
        mock_redis_client.setex.assert_not_called()

        # Test Cache Invalidation on add_memory
        mock_redis_client.scan_iter.return_value = [b'search:cached data:2']
        mock_redis_client.delete.reset_mock() # Reset delete mock

        response_add_new = await ac.post("/api/v1/memory/add", json={"content": "New memory to invalidate cache.", "tags": ["new"]})
        assert response_add_new.status_code == 201

        mock_redis_client.scan_iter.assert_called_with("search:*")
        mock_redis_client.delete.assert_called_with(b'search:cached data:2')

        # Test Cache Invalidation on redact_memory (placeholder)
        pass

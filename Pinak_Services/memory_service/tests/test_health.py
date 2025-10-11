"""Tests for health check endpoints."""

import json
import pytest
from httpx import AsyncClient, ASGITransport
from asgi_lifespan import LifespanManager

from app.main import app
from app.api.v1.endpoints import get_memory_service
from app.services.memory_service import MemoryService


@pytest.fixture(autouse=True)
def _configure_environment(monkeypatch):
    monkeypatch.setenv("PINAK_JWT_SECRET", "test-secret")
    monkeypatch.setenv("PINAK_EMBEDDING_BACKEND", "dummy")


@pytest.fixture
def memory_service(tmp_path):
    config = {
        "embedding_model": "dummy",
        "data_root": str(tmp_path / "data"),
        "redis_host": "localhost",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    service = MemoryService(config_path=str(config_path))
    return service


@pytest.fixture
def test_app(memory_service):
    app.dependency_overrides[get_memory_service] = lambda: memory_service
    yield app
    app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_health_endpoint_exists(test_app):
    """Test that /health endpoint exists and returns 200."""
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_health_endpoint_returns_json(test_app):
    """Test that /health endpoint returns JSON response."""
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/health")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"


@pytest.mark.asyncio
async def test_health_endpoint_has_status_field(test_app):
    """Test that /health response includes status field."""
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/health")
    data = response.json()
    assert "status" in data
    assert data["status"] in ["healthy", "unhealthy", "degraded"]


@pytest.mark.asyncio
async def test_health_endpoint_checks_service(test_app):
    """Test that /health response includes service check."""
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/health")
    data = response.json()
    assert "service" in data
    assert data["service"] == "memory-service"


@pytest.mark.asyncio
async def test_health_endpoint_has_version(test_app):
    """Test that /health response includes version information."""
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/health")
    data = response.json()
    assert "version" in data
    assert isinstance(data["version"], str)


@pytest.mark.asyncio
async def test_health_endpoint_has_timestamp(test_app):
    """Test that /health response includes timestamp."""
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/health")
    data = response.json()
    assert "timestamp" in data
    assert isinstance(data["timestamp"], str)


@pytest.mark.asyncio
async def test_health_endpoint_checks_components(test_app):
    """Test that /health response includes component health checks."""
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/health")
    data = response.json()
    assert "checks" in data
    assert isinstance(data["checks"], dict)


@pytest.mark.asyncio
async def test_health_endpoint_checks_memory_service(test_app):
    """Test that /health checks memory service availability."""
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/health")
    data = response.json()
    assert "checks" in data
    checks = data["checks"]
    assert "memory_service" in checks
    assert checks["memory_service"]["status"] in ["ok", "error"]


@pytest.mark.asyncio
async def test_health_endpoint_does_not_require_auth(test_app):
    """Test that /health endpoint does not require authentication."""
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            # No Authorization header
            response = await client.get("/health")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_readiness_endpoint_exists(test_app):
    """Test that /health/ready endpoint exists."""
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/health/ready")
    assert response.status_code in [200, 503]


@pytest.mark.asyncio
async def test_liveness_endpoint_exists(test_app):
    """Test that /health/live endpoint exists."""
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/health/live")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_liveness_endpoint_returns_simple_response(test_app):
    """Test that /health/live returns simple response."""
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/health/live")
    data = response.json()
    assert "status" in data
    assert data["status"] == "alive"


@pytest.mark.asyncio
async def test_health_endpoint_handles_errors_gracefully(test_app):
    """Test that /health endpoint handles component errors gracefully."""
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/health")
    # Should always return 200 even if some checks fail
    assert response.status_code == 200
    data = response.json()
    # But status might be unhealthy or degraded
    assert data["status"] in ["healthy", "unhealthy", "degraded"]

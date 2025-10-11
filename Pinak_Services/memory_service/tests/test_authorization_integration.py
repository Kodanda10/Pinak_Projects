"""Integration tests for role-based authorization with API endpoints."""

import datetime
import json
import jwt
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


def _issue_token_with_role(tenant: str, project: str, role: str, subject: str = "tester") -> str:
    """Issue a JWT token with a specific role."""
    payload = {
        "sub": subject,
        "tenant": tenant,
        "project_id": project,
        "roles": [role],
        "iat": datetime.datetime.now(datetime.UTC),
        "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=5),
    }
    return jwt.encode(payload, "test-secret", algorithm="HS256")


@pytest.mark.asyncio
async def test_admin_can_add_memory(test_app):
    """Test that admin role can add memory."""
    token = _issue_token_with_role("tenant-a", "project-1", "admin")
    headers = {"Authorization": f"Bearer {token}"}
    
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/memory/add",
                headers=headers,
                json={"content": "Admin test", "tags": ["test"]},
            )
    assert response.status_code == 201


@pytest.mark.asyncio
async def test_user_can_add_memory(test_app):
    """Test that user role can add memory."""
    token = _issue_token_with_role("tenant-a", "project-1", "user")
    headers = {"Authorization": f"Bearer {token}"}
    
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/memory/add",
                headers=headers,
                json={"content": "User test", "tags": ["test"]},
            )
    assert response.status_code == 201


@pytest.mark.asyncio
async def test_guest_can_read_memory(test_app, memory_service):
    """Test that guest role can read memory."""
    # First add memory as admin
    admin_token = _issue_token_with_role("tenant-a", "project-1", "admin")
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    
    # Then read as guest
    guest_token = _issue_token_with_role("tenant-a", "project-1", "guest")
    guest_headers = {"Authorization": f"Bearer {guest_token}"}
    
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            # Add as admin
            await client.post(
                "/api/v1/memory/add",
                headers=admin_headers,
                json={"content": "Guest read test", "tags": ["test"]},
            )
            
            # Read as guest
            response = await client.get(
                "/api/v1/memory/search?query=test&limit=5",
                headers=guest_headers,
            )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_admin_can_read_events(test_app):
    """Test that admin role can read events."""
    token = _issue_token_with_role("tenant-a", "project-1", "admin")
    headers = {"Authorization": f"Bearer {token}"}
    
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get(
                "/api/v1/memory/events",
                headers=headers,
            )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_user_can_write_events(test_app):
    """Test that user role can write events."""
    token = _issue_token_with_role("tenant-a", "project-1", "user")
    headers = {"Authorization": f"Bearer {token}"}
    
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/memory/event",
                headers=headers,
                json={
                    "event_type": "test_event",
                    "description": "User event test",
                    "metadata": {"key": "value"}
                },
            )
    assert response.status_code == 201


@pytest.mark.asyncio
async def test_guest_can_read_events(test_app):
    """Test that guest role can read events."""
    token = _issue_token_with_role("tenant-a", "project-1", "guest")
    headers = {"Authorization": f"Bearer {token}"}
    
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get(
                "/api/v1/memory/events",
                headers=headers,
            )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_multiple_roles_grant_union_of_permissions(test_app):
    """Test that multiple roles grant union of permissions."""
    # Token with both user and service roles
    payload = {
        "sub": "tester",
        "tenant": "tenant-a",
        "project_id": "project-1",
        "roles": ["user", "service"],
        "iat": datetime.datetime.now(datetime.UTC),
        "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=5),
    }
    token = jwt.encode(payload, "test-secret", algorithm="HS256")
    headers = {"Authorization": f"Bearer {token}"}
    
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/memory/add",
                headers=headers,
                json={"content": "Multi-role test", "tags": ["test"]},
            )
    assert response.status_code == 201

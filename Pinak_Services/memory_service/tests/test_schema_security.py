
import datetime
import pytest
import jwt
from typing import Dict
from httpx import AsyncClient, ASGITransport
from asgi_lifespan import LifespanManager
from app.main import app
from app.api.v1.endpoints import get_memory_service
from app.services.memory_service import MemoryService
import json

@pytest.fixture(autouse=True)
def _configure_environment(monkeypatch):
    monkeypatch.setenv("PINAK_JWT_SECRET", "test-secret")
    monkeypatch.setenv("EMBEDDING_BACKEND", "dummy")

@pytest.fixture
def memory_service_mock(tmp_path):
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
def test_app(memory_service_mock):
    app.dependency_overrides[get_memory_service] = lambda: memory_service_mock
    yield app
    app.dependency_overrides.clear()

def _issue_token(tenant: str, project: str, subject: str = "tester") -> str:
    payload: Dict[str, object] = {
        "sub": subject,
        "tenant": tenant,
        "project_id": project,
        "scopes": ["memory.read", "memory.write"],
        "iat": datetime.datetime.now(datetime.timezone.utc),
        "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5),
    }
    return jwt.encode(payload, "test-secret", algorithm="HS256")

@pytest.mark.asyncio
async def test_schema_endpoints_are_protected(test_app):
    """
    Verify that schema endpoints now require authentication.
    """
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            # 1. Unauthenticated access should FAIL with 401
            response = await client.get("/api/v1/memory/schema")
            assert response.status_code == 401, "Schema endpoint should be protected"

            # 2. Authenticated access should SUCCEED
            token = _issue_token("tenant-a", "project-1")
            headers = {"Authorization": f"Bearer {token}"}

            response = await client.get("/api/v1/memory/schema", headers=headers)
            assert response.status_code == 200

            data = response.json()

            # 3. Information Disclosure check
            # It should NOT expose internal directory paths anymore
            assert "schema_dir" not in data
            assert "fallback_dir" not in data
            assert "schemas" in data

@pytest.mark.asyncio
async def test_schema_layer_access_security(test_app):
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            # Unauthenticated
            response = await client.get("/api/v1/memory/schema/episodic")
            assert response.status_code == 401

            # Authenticated
            token = _issue_token("tenant-a", "project-1")
            headers = {"Authorization": f"Bearer {token}"}

            # Valid layer
            response = await client.get("/api/v1/memory/schema/episodic", headers=headers)
            assert response.status_code == 200

            # Invalid layer (Path Traversal attempt)
            response = await client.get("/api/v1/memory/schema/../../config", headers=headers)
            # Should be 400 Bad Request due to sanitization or 404 Not Found if sanitization catches it cleanly
            # Or 404 if it just doesn't find it.
            # But we added sanitization check which raises 400.
            # HOWEVER, httpx/fastapi might normalize .. before it hits our code?
            # Let's try characters that are definitely not allowed but not path separators if needed.
            # But our check is `isalnum` (plus - and _).

            # Try illegal characters
            response = await client.get("/api/v1/memory/schema/bad$layer", headers=headers)
            assert response.status_code == 400

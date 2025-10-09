import datetime
import json
import os
from typing import Dict

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


def _issue_token(tenant: str, project: str, subject: str = "tester") -> str:
    payload: Dict[str, object] = {
        "sub": subject,
        "tenant": tenant,
        "project_id": project,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=5),
    }
    return jwt.encode(payload, "test-secret", algorithm="HS256")


@pytest.mark.asyncio
async def test_missing_token_is_rejected(test_app):
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/api/v1/memory/episodic/list")
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_memory_isolation_between_tenants(test_app, memory_service):
    token_a = _issue_token("tenant-a", "project-1")
    token_b = _issue_token("tenant-b", "project-1")

    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            await client.post("/api/v1/memory/add", headers=headers_a, json={"content": "alpha memory", "tags": ["alpha"]})
            await client.post("/api/v1/memory/add", headers=headers_b, json={"content": "beta memory", "tags": ["beta"]})

            search_a = await client.get("/api/v1/memory/search", headers=headers_a, params={"query": "alpha"})
            search_b = await client.get("/api/v1/memory/search", headers=headers_b, params={"query": "beta"})

    assert search_a.status_code == 200
    assert search_b.status_code == 200
    assert all(result["tenant"] == "tenant-a" for result in search_a.json())
    assert all(result["tenant"] == "tenant-b" for result in search_b.json())


@pytest.mark.asyncio
async def test_hash_chained_audit_log(test_app, memory_service):
    token = _issue_token("tenant-log", "project-log")
    headers = {"Authorization": f"Bearer {token}"}

    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            for step in range(3):
                await client.post(
                    "/api/v1/memory/event",
                    headers=headers,
                    json={"type": "audit", "step": step},
                )

    base = memory_service._store_dir("tenant-log", "project-log")
    events_folder = os.path.join(base, "events")
    event_files = sorted(f for f in os.listdir(events_folder) if f.startswith("events_"))
    assert event_files, "expected at least one event log file"

    latest_file = os.path.join(events_folder, event_files[-1])
    with open(latest_file, "r", encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh if line.strip()]

    assert len(records) == 3
    assert records[0]["prev_hash"] is None
    for idx in range(1, len(records)):
        assert records[idx]["prev_hash"] == records[idx - 1]["hash"]
        expected_hash = memory_service._compute_audit_hash({k: v for k, v in records[idx].items() if k != "hash"})
        assert records[idx]["hash"] == expected_hash


@pytest.mark.asyncio
async def test_session_and_working_memory_are_scoped(test_app):
    token_a = _issue_token("tenant-a", "project-1")
    token_b = _issue_token("tenant-b", "project-1")
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            await client.post(
                "/api/v1/memory/session/add",
                headers=headers_a,
                json={"session_id": "conversation", "content": "hello"},
            )
            await client.post(
                "/api/v1/memory/session/add",
                headers=headers_b,
                json={"session_id": "conversation", "content": "hola"},
            )

            await client.post(
                "/api/v1/memory/working/add",
                headers=headers_a,
                json={"content": "scratchpad"},
            )
            await client.post(
                "/api/v1/memory/working/add",
                headers=headers_b,
                json={"content": "notepad"},
            )

            session_a = await client.get(
                "/api/v1/memory/session/list",
                headers=headers_a,
                params={"session_id": "conversation"},
            )
            session_b = await client.get(
                "/api/v1/memory/session/list",
                headers=headers_b,
                params={"session_id": "conversation"},
            )

            working_a = await client.get("/api/v1/memory/working/list", headers=headers_a)
            working_b = await client.get("/api/v1/memory/working/list", headers=headers_b)

    assert session_a.status_code == 200
    assert session_b.status_code == 200
    assert working_a.status_code == 200
    assert working_b.status_code == 200

    assert all(entry["tenant"] == "tenant-a" for entry in session_a.json())
    assert all(entry["tenant"] == "tenant-b" for entry in session_b.json())
    assert all(entry["tenant"] == "tenant-a" for entry in working_a.json())
    assert all(entry["tenant"] == "tenant-b" for entry in working_b.json())


@pytest.mark.asyncio
async def test_invalid_token_is_rejected(test_app):
    headers = {"Authorization": "Bearer invalid"}
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/api/v1/memory/episodic/list", headers=headers)
    assert response.status_code == 401

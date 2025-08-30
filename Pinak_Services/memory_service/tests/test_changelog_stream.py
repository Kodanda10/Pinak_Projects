import json
import os

import pytest
from app.main import app
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("USE_MOCK_EMBEDDINGS", "true")


@pytest.mark.asyncio
async def test_changelog_create_and_redact(client_and_memory_service):
    """Test changelog creation and redaction functionality."""
    client, service = client_and_memory_service

    # add creates a changelog entry (best-effort)
    r = await client.post(
        "/api/v1/memory/add", json={"content": "to redact", "tags": []}
    )
    assert r.status_code == 201
    mid = r.json()["id"]
    # redact
    rr = await client.post(
        "/api/v1/memory/changelog/redact", json={"memory_id": mid, "reason": "cleanup"}
    )
    assert rr.status_code == 200
    # list changelog
    cl = await client.get("/api/v1/memory/changelog")
    assert cl.status_code == 200
    body = cl.json()
    assert any(e.get("change_type") == "create" for e in body)
    assert any(
        e.get("change_type") == "redact" and e.get("target_id") == mid for e in body
    )

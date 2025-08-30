import datetime
import os

import pytest
from app.main import app
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

os.environ.setdefault("USE_MOCK_EMBEDDINGS", "true")


@pytest.mark.asyncio
async def test_session_add_ttl_and_list_paging():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            sid = "s-paging"
            # Insert 3 items
            for i in range(3):
                r = await ac.post(
                    "/api/v1/memory/session/add",
                    json={"session_id": sid, "content": f"c{i}"},
                )
                assert r.status_code == 201
            # Page 2 items
            lst = await ac.get(
                "/api/v1/memory/session/list",
                params={"session_id": sid, "limit": 2, "offset": 1},
            )
            assert lst.status_code == 200
            data = lst.json()
            assert len(data) == 2


@pytest.mark.asyncio
async def test_working_ttl_respected():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            # expired record
            past = (datetime.datetime.utcnow() - datetime.timedelta(seconds=5)).isoformat()
            r1 = await ac.post(
                "/api/v1/memory/working/add",
                json={"content": "old", "expires_at": past},
            )
            assert r1.status_code == 201
            # valid record
            r2 = await ac.post(
                "/api/v1/memory/working/add",
                json={"content": "new", "ttl_seconds": 100},
            )
            assert r2.status_code == 201
            lst = await ac.get("/api/v1/memory/working/list")
            assert lst.status_code == 200
            contents = [x.get("content") for x in lst.json()]
            assert "new" in contents and "old" not in contents

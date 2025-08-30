import pytest
from app.main import app
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

pytestmark = pytest.mark.asyncio


async def test_audit_verify_events():
    pid = "Pnk-AV"
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.post(
                "/api/v1/memory/event",
                headers={"X-Pinak-Project": pid},
                json={
                    "type": "gov_audit",
                    "path": "/x",
                    "method": "POST",
                    "status": 200,
                },
            )
            assert r.status_code == 201
            v = await ac.get(
                "/api/v1/memory/audit/verify",
                headers={"X-Pinak-Project": pid},
                params={"kind": "events"},
            )
            assert v.status_code == 200
            data = v.json()
            assert data.get("ok") is True
            assert data.get("count", 0) >= 1

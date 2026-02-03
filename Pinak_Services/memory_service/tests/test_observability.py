import pytest
import pytest_asyncio
from app.core.database import DatabaseManager

pytestmark = pytest.mark.asyncio

@pytest_asyncio.fixture
async def db():
    dm = DatabaseManager(":memory:")
    await dm.init_db()
    return dm

async def test_upsert_agent_and_list(db):
    res = await db.upsert_agent(
        agent_id="agent-1",
        client_name="codex",
        status="active",
        tenant="default",
        project_id="pinak-memory",
        hostname="host",
        pid="123",
        meta={"version": "1.0"},
    )
    assert res["agent_id"] == "agent-1"

    # Upsert same agent should update last_seen and status
    res2 = await db.upsert_agent(
        agent_id="agent-1",
        client_name="codex",
        status="idle",
        tenant="default",
        project_id="pinak-memory",
        hostname="host",
        pid="123",
        meta={"version": "1.1"},
    )
    assert res2["status"] == "idle"

    agents = await db.list_agents("default", "pinak-memory", limit=10)
    assert len(agents) == 1
    assert agents[0]["client_name"] == "codex"
    assert agents[0]["status"] == "idle"
    assert agents[0]["meta"]["version"] == "1.1"


async def test_access_event_log(db):
    event = await db.add_access_event(
        event_type="read",
        status="ok",
        tenant="default",
        project_id="pinak-memory",
        agent_id="agent-2",
        client_name="gemini",
        target_layer="semantic",
        query="hello world",
        result_count=2,
    )
    assert event["event_type"] == "read"

    rows = await db.list_access_events("default", "pinak-memory", limit=10)
    assert len(rows) == 1
    assert rows[0]["client_name"] == "gemini"
    assert rows[0]["result_count"] == 2

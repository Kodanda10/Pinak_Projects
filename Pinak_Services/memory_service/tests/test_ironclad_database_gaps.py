import pytest
import pytest_asyncio
from app.core.database import DatabaseManager

pytestmark = pytest.mark.asyncio

@pytest_asyncio.fixture
async def db():
    # Use in-memory or global test DB
    dm = DatabaseManager(":memory:")
    await dm.init_db()
    return dm

async def test_database_procedural_gap(db):
    await db.add_procedural("fix logic", ["step1", "step2"], "t1", "p1", trigger="error")
    results = await db.search_keyword("fix", "t1", "p1")
    
    found = False
    for r in results:
        if r["type"] == "procedural":
            assert "steps" in r
            assert r["steps"] == ["step1", "step2"]
            found = True
    assert found

async def test_database_update_delete_layer_errors(db):
    with pytest.raises(ValueError, match="Invalid layer"):
        await db.update_memory("ghost", "id", {}, "t", "p")
    
    with pytest.raises(ValueError, match="Invalid layer"):
        await db.delete_memory("ghost", "id", "t", "p")

async def test_database_episodic_search_parsing(db):
    await db.add_episodic("trip report", "t1", "p1", plan=["go"], tool_logs=[{"t": 1}])
    results = await db.search_keyword("trip", "t1", "p1")
    
    found = False
    for r in results:
        if r["type"] == "episodic":
            assert r["plan"] == ["go"]
            assert r["tool_logs"] == [{"t": 1}]
            found = True
    assert found

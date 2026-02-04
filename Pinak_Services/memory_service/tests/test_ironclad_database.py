import pytest
import pytest_asyncio
import os
import json
import sqlite3
from unittest.mock import patch
from app.core.database import DatabaseManager

# Mark all tests in this module as async
pytestmark = pytest.mark.asyncio

@pytest_asyncio.fixture
async def db():
    # Uses the global test DB configured in conftest.py
    dm = DatabaseManager(":memory:") # Path ignored by async_db logic currently
    await dm.init_db()
    return dm

async def test_add_all_memory_layers(db):
    res_ep = await db.add_episodic("experience", "t1", "p1", goal="be cool", plan=["step1"], tool_logs=[{"tool": "cmd"}])
    assert res_ep["id"] is not None
    
    res_pr = await db.add_procedural("skill", ["step1"], "t1", "p1", trigger="trigger")
    assert res_pr["id"] is not None
    
    res_rag = await db.add_rag("query", "source", "content", "t1", "p1")
    assert res_rag["id"] is not None

async def test_get_memory_invalid_layer(db):
    assert await db.get_memory("invalid", "id", "t", "p") is None

async def test_get_episodic_json_loading(db):
    res = await db.add_episodic("content", "t1", "p1", plan=["a"], tool_logs=[{"b": 1}])
    mid = res["id"]
    
    mem = await db.get_memory("episodic", mid, "t1", "p1")
    assert mem["plan"] == ["a"]
    assert mem["tool_logs"] == [{"b": 1}]

async def test_search_keyword_all_layers(db):
    await db.add_semantic("semantic search target", [], "t1", "p1", 1)
    await db.add_episodic("episodic search target", "t1", "p1")
    await db.add_procedural("procedural search", ["target"], "t1", "p1")
    
    results = await db.search_keyword("target", "t1", "p1")
    types = [r["type"] for r in results]
    assert "semantic" in types
    assert "episodic" in types
    assert "procedural" in types

async def test_update_delete_invalid_layer(db):
    with pytest.raises(ValueError, match="Invalid layer"):
        await db.update_memory("invalid", "id", {}, "t", "p")
    with pytest.raises(ValueError, match="Invalid layer"):
        await db.delete_memory("invalid", "id", "t", "p")

import pytest
import os
import json
import sqlite3
from unittest.mock import patch
from app.core.database import DatabaseManager

@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    return DatabaseManager(db_path)

def test_database_init_failure():
    with patch("os.makedirs", side_effect=PermissionError("Perm error")):
        with pytest.raises(PermissionError):
            DatabaseManager("/root/protected.db")

def test_get_cursor_rollback(db):
    # Ensure table exists first
    with db.get_cursor() as cur:
        cur.execute("CREATE TABLE test_rollback (name TEXT)")
    
    try:
        with db.get_cursor() as cur:
            cur.execute("INSERT INTO test_rollback (name) VALUES (?)", ("should_be_rolled_back",))
            raise RuntimeError("Force Rollback")
    except RuntimeError:
        pass
    
    # Verify insert was rolled back
    with db.get_cursor() as cur:
        cur.execute("SELECT count(*) FROM test_rollback WHERE name = ?", ("should_be_rolled_back",))
        assert cur.fetchone()[0] == 0

def test_add_all_memory_layers(db):
    res_ep = db.add_episodic("experience", "t1", "p1", goal="be cool", plan=["step1"], tool_logs=[{"tool": "cmd"}])
    assert res_ep["id"] is not None
    
    res_pr = db.add_procedural("skill", ["step1"], "t1", "p1", trigger="trigger")
    assert res_pr["id"] is not None
    
    res_rag = db.add_rag("query", "source", "content", "t1", "p1")
    assert res_rag["id"] is not None

def test_get_memory_invalid_layer(db):
    assert db.get_memory("invalid", "id", "t", "p") is None

def test_get_episodic_json_loading(db):
    res = db.add_episodic("content", "t1", "p1", plan=["a"], tool_logs=[{"b": 1}])
    mid = res["id"]
    
    mem = db.get_memory("episodic", mid, "t1", "p1")
    assert mem["plan"] == ["a"]
    assert mem["tool_logs"] == [{"b": 1}]

def test_search_keyword_all_layers(db):
    db.add_semantic("semantic search target", [], "t1", "p1", 1)
    db.add_episodic("episodic search target", "t1", "p1")
    db.add_procedural("procedural search", ["target"], "t1", "p1")
    
    # Needs a moment or explicit commit? SQLite FTS is usually immediate with triggers.
    results = db.search_keyword("target", "t1", "p1")
    types = [r["type"] for r in results]
    assert "semantic" in types
    assert "episodic" in types
    assert "procedural" in types

def test_chaos_database_corruption(tmp_path):
    db_path = tmp_path / "corrupt.db"
    db_path.write_bytes(b"NOT A SQLITE FILE" * 100)
    
    # Re-init should handle it or at least not fail silently if it attempts to write more
    # Actually DatabaseManager._init_db might fail on PRAGMA if it's invalid.
    with pytest.raises(sqlite3.DatabaseError):
        dm = DatabaseManager(str(db_path))
        with dm.get_cursor() as cur:
            cur.execute("SELECT * FROM memories_semantic")

def test_update_delete_invalid_layer(db):
    with pytest.raises(ValueError, match="Invalid layer"):
        db.update_memory("invalid", "id", {}, "t", "p")
    with pytest.raises(ValueError, match="Invalid layer"):
        db.delete_memory("invalid", "id", "t", "p")

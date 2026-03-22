import pytest
from app.core.database import DatabaseManager

def test_sql_injection_mitigation(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = DatabaseManager(db_path)
    res = db.add_semantic("test content", [], "t1", "p1", 1)
    mem_id = res["id"]

    # Attempt SQL injection through dictionary keys
    updates = {
        "content": "new content",
        "agent_id = 'hacker', client_id": "hacker_client"
    }

    with pytest.raises(ValueError, match="Invalid update key"):
        db.update_memory("semantic", mem_id, updates, "t1", "p1")

    mem = db.get_memory("semantic", mem_id, "t1", "p1")
    assert mem["content"] == "test content" # Check that content did not update.
    assert mem.get("agent_id") is None # Ensure agent_id was not modified
    assert mem.get("client_id") is None # Ensure client_id was not modified

def test_valid_update(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = DatabaseManager(db_path)
    res = db.add_semantic("test content", [], "t1", "p1", 1)
    mem_id = res["id"]

    updates = {
        "content": "new content",
        "agent_id": "test_agent"
    }

    db.update_memory("semantic", mem_id, updates, "t1", "p1")
    mem = db.get_memory("semantic", mem_id, "t1", "p1")

    assert mem["content"] == "new content"
    assert mem["agent_id"] == "test_agent"

def test_invalid_type_key(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = DatabaseManager(db_path)
    res = db.add_semantic("test content", [], "t1", "p1", 1)
    mem_id = res["id"]

    updates = {
        123: "new content",
    }

    with pytest.raises(ValueError, match="Invalid update key"):
        db.update_memory("semantic", mem_id, updates, "t1", "p1")

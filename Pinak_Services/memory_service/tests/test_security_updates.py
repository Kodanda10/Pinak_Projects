import pytest
from app.core.database import DatabaseManager

def test_database_manager_update_memory_sql_injection(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = DatabaseManager(db_path)

    # Add a semantic memory
    res = db.add_semantic(
        content="test", tags=["a"], tenant="t1", project_id="p1", embedding_id=1
    )
    mid = res["id"]

    # Attempt SQL injection via keys
    updates = {
        "content = 'hacked' --": "value",
        "tags": ["b"]
    }

    # This should not raise sqlite3.OperationalError: near "=": syntax error
    # and "content = 'hacked' --" should be ignored
    success = db.update_memory("semantic", mid, updates, "t1", "p1")
    assert success is True

    # Verify the item wasn't hacked
    item = db.get_memory("semantic", mid, "t1", "p1")
    assert item["tags"] == ["b"]
    assert item["content"] == "test"

def test_database_manager_update_memory_empty(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = DatabaseManager(db_path)

    # Add a semantic memory
    res = db.add_semantic(
        content="test", tags=["a"], tenant="t1", project_id="p1", embedding_id=1
    )
    mid = res["id"]

    # Update with empty safe updates
    updates = {
        "invalid!": "value"
    }
    success = db.update_memory("semantic", mid, updates, "t1", "p1")
    assert success is False

import pytest
from app.core.database import DatabaseManager

def test_sql_injection_in_update_keys(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = DatabaseManager(db_path)

    res = db.add_semantic("hello", [], "t1", "p1", 1)
    mid = res["id"]

    # Attempt SQL injection via update keys
    with pytest.raises(ValueError, match="Invalid update key"):
        db.update_memory("semantic", mid, {"content = 'injected', content": "value"}, "t1", "p1")

def test_valid_update_key(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = DatabaseManager(db_path)

    res = db.add_semantic("hello", [], "t1", "p1", 1)
    mid = res["id"]

    db.update_memory("semantic", mid, {"content": "world"}, "t1", "p1")
    item = db.get_memory("semantic", mid, "t1", "p1")
    assert item["content"] == "world"

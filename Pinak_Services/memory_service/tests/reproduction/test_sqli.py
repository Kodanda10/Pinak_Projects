import pytest
import sqlite3
import os
from app.core.database import DatabaseManager

def test_update_memory_sqli(tmp_path):
    db_path = tmp_path / "test_sqli.db"
    db = DatabaseManager(str(db_path))

    res1 = db.add_semantic("hello world", ["tag1"], "tenant1", "project1", 1)

    # Try valid update
    db.update_memory("semantic", res1["id"], {"content": "valid update"}, "tenant1", "project1")
    assert db.get_memory("semantic", res1["id"], "tenant1", "project1")["content"] == "valid update"

    # Try SQL injection in update keys
    with pytest.raises(ValueError, match="Invalid column name: content = 'hacked' --"):
        db.update_memory("semantic", res1["id"], {"content = 'hacked' --": "someval"}, "tenant1", "project1")

    with pytest.raises(ValueError, match="Invalid column name: content` = 'hacked'"):
        db.update_memory("semantic", res1["id"], {"content` = 'hacked'": "someval"}, "tenant1", "project1")

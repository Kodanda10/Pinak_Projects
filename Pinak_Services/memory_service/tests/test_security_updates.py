import pytest
from app.core.database import DatabaseManager

def test_sql_injection_in_update_memory(tmp_path):
    db_path = str(tmp_path / "memory.db")
    db = DatabaseManager(db_path)

    # create a memory to update
    db.add_semantic("test", [], "tenant1", "proj1", 1)

    # Try an injection in keys
    malicious_updates = {"content = ?; DROP TABLE memories_semantic; --": "hacked"}

    with pytest.raises(ValueError):
        db.update_memory("semantic", "some_id", malicious_updates, "tenant1", "proj1")

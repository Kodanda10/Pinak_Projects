import pytest
from app.services.memory_service import MemoryService
from app.core.database import DatabaseManager

def test_memory_service_update_whitelist(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"embedding_model": "dummy", "data_root": "' + str(tmp_path) + '"}')
    svc = MemoryService(config_path=str(config_path))

    # Create memory
    res = svc.add_memory(type('obj', (object,), {'content': 'hello', 'tags': []}), "t1", "p1")

    # Update with malicious keys
    updates = {"content": "safe", "id": "malicious", "tenant": "malicious", "project_id": "malicious", "drop_table": "yes"}
    svc.update_memory("semantic", res.id, updates, "t1", "p1")

    # Verify only content was updated
    item = svc.db.get_memory("semantic", res.id, "t1", "p1")
    assert item["content"] == "safe"
    assert item["id"] == res.id
    assert item["tenant"] == "t1"
    assert "drop_table" not in item

def test_database_manager_update_sql_injection(tmp_path):
    db_path = tmp_path / "memory.db"
    db = DatabaseManager(str(db_path))

    # Try updating with malicious key
    with pytest.raises(ValueError, match="Invalid column name"):
        db.update_memory("semantic", "some_id", {"malicious_col = 1; --": "value"}, "t1", "p1")

    # Try updating with non-string key
    with pytest.raises(ValueError, match="Invalid column name"):
        db.update_memory("semantic", "some_id", {1: "value"}, "t1", "p1")

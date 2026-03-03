import pytest
from app.core.database import DatabaseManager

@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test_sqli.db")
    return DatabaseManager(db_path)

def test_database_update_memory_sqli_prevention(db):
    """
    Tests that DatabaseManager.update_memory prevents SQL injection
    by validating that all keys in the updates dictionary are valid Python identifiers.
    """
    # Create a dummy memory item to attempt to update
    res = db.add_semantic(
        content="safe content",
        tags=["safe"],
        tenant="t1",
        project_id="p1",
        embedding_id=1
    )
    memory_id = res["id"]

    # Attempt to inject SQL via a malicious key
    malicious_updates = {
        "content = 'hacked'; DROP TABLE memories_semantic; --": "value"
    }

    with pytest.raises(ValueError) as excinfo:
        db.update_memory("semantic", memory_id, malicious_updates, "t1", "p1")

    assert "Invalid column name" in str(excinfo.value)

    # Verify the table still exists and the original content is intact
    memory = db.get_memory("semantic", memory_id, "t1", "p1")
    assert memory is not None
    assert memory["content"] == "safe content"

def test_database_update_memory_invalid_key_type(db):
    """
    Tests that DatabaseManager.update_memory prevents updates with non-string keys.
    """
    res = db.add_semantic(
        content="safe content",
        tags=["safe"],
        tenant="t1",
        project_id="p1",
        embedding_id=1
    )
    memory_id = res["id"]

    invalid_updates = {
        123: "value"
    }

    with pytest.raises(ValueError) as excinfo:
        db.update_memory("semantic", memory_id, invalid_updates, "t1", "p1")

    assert "Invalid column name: 123" in str(excinfo.value)

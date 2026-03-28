import pytest
import os
from app.core.database import DatabaseManager

def test_sql_injection_in_update_memory_keys():
    """
    Test that malicious dictionary keys containing SQL injection payloads
    are rejected by DatabaseManager.update_memory.
    """
    # Use an in-memory database for testing
    # Note: sqlite3 treats every connect() to :memory: as a NEW database unless
    # sharing cache. For testing, we use a temp file to ensure persistence across cursors.
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = DatabaseManager(db_path)

        # Create a dummy memory entry to update
        db.add_semantic(
            content="Test memory",
            tags=["test"],
            tenant="tenant1",
            project_id="proj1",
            embedding_id=1
        )

        # Get the inserted memory ID
        with db.get_cursor() as conn:
            row = conn.execute("SELECT id FROM memories_semantic LIMIT 1").fetchone()
            memory_id = row[0]

        # Valid update
        success = db.update_memory(
            layer="semantic",
            memory_id=memory_id,
            updates={"content": "Updated content"},
            tenant="tenant1",
            project_id="proj1"
        )
        assert success is True

        # SQL Injection payload in the key
        malicious_key = "content = 'hacked', tags = '[]' WHERE id = 1; DROP TABLE memories_semantic; --"
        with pytest.raises(ValueError, match="Invalid column name for update"):
            db.update_memory(
                layer="semantic",
                memory_id=memory_id,
                updates={malicious_key: "hacked"},
                tenant="tenant1",
                project_id="proj1"
            )

        # Another payload
        malicious_key_2 = "content = 'pwned' --"
        with pytest.raises(ValueError, match="Invalid column name for update"):
            db.update_memory(
                layer="semantic",
                memory_id=memory_id,
                updates={malicious_key_2: "pwned"},
                tenant="tenant1",
                project_id="proj1"
            )

        # Payload with non-string key
        with pytest.raises(ValueError, match="Invalid column name for update"):
            db.update_memory(
                layer="semantic",
                memory_id=memory_id,
                updates={123: "pwned"},
                tenant="tenant1",
                project_id="proj1"
            )
    finally:
        os.unlink(db_path)

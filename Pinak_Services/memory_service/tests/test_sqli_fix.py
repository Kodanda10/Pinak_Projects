import os
import time
import pytest
import json
import uuid
from app.core.database import DatabaseManager

def test_sqli_prevention_in_update_memory(tmp_path):
    db_path = str(tmp_path / "test_db.db")
    db = DatabaseManager(db_path)
    db._init_db()

    with db.get_cursor() as conn:
        conn.execute("INSERT INTO memories_semantic (id, tenant, project_id, content, created_at, tags) VALUES ('mem1', 'tenant1', 'proj1', 'test', ?, ?)", (int(time.time()), json.dumps([])))

    updates = {
        "content = 'hacked' --": "ignored",
    }

    with pytest.raises(ValueError, match="Invalid update key"):
        db.update_memory("semantic", "mem1", updates, "tenant1", "proj1")

    with db.get_cursor() as conn:
        cur = conn.execute("SELECT content FROM memories_semantic WHERE id = 'mem1'")
        assert cur.fetchone()[0] == "test"


import pytest
import sqlite3
import os
from app.core.database import DatabaseManager

def test_sql_injection_update_mitigated():
    db_path = "test_sqli_mitigated.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    try:
        db = DatabaseManager(db_path)

        # Setup: Add a memory for Tenant A
        with db.get_cursor() as conn:
            conn.execute("INSERT INTO memories_semantic (id, content, tenant, project_id, created_at) VALUES ('mem1', 'original', 'tenantA', 'projA', 'now')")

        updates = {
            "content": "updated",
            "id = id --": "ignored"
        }

        # Expect ValueError now
        try:
            db.update_memory("semantic", "mem1", updates, "tenantA", "projA")
            assert False, "Should have raised ValueError for invalid column name"
        except ValueError as e:
            assert "Invalid column name" in str(e)
        except sqlite3.ProgrammingError:
            assert False, "Should have been caught by input validation before hitting SQL"
    finally:
        if os.path.exists(db_path):
            os.remove(db_path)

if __name__ == "__main__":
    test_sql_injection_update_mitigated()
    print("Test passed: SQL injection attempt blocked with ValueError.")

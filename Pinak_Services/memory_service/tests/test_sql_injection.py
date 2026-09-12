import pytest
from app.core.database import DatabaseManager

@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    return DatabaseManager(db_path)

def test_update_memory_sql_injection(db):
    res = db.add_semantic("semantic search target", [], "t1", "p1", 1)
    memory_id = res["id"]

    # Simulate SQL injection in column name
    # The vulnerability is in app/core/database.py:
    # set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])
    # By passing a key that contains a quote, we can break the query.

    # We will try to add a malicious key
    malicious_updates = {"\"content\" = 'hacked' --": "value"}

    try:
        # If vulnerable, this will raise a sqlite3.OperationalError
        db.update_memory("semantic", memory_id, malicious_updates, "t1", "p1")
        vulnerable = False
    except Exception as e:
        if "Incorrect number of bindings supplied" in str(e):
            vulnerable = True
            print(f"Injection successful! Error: {e}")
        else:
            vulnerable = False
            print(f"Injection failed (secure)! Error: {e}")

    assert not vulnerable, "SQL Injection vulnerability exists in update_memory"

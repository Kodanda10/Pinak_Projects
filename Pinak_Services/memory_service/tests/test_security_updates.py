import pytest
from app.core.database import DatabaseManager
import sqlite3

@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "test.db")
    return DatabaseManager(db_path)

def test_update_memory_sql_injection_defense(db):
    db.add_episodic("test content", "t1", "p1")

    # "bad_column" passes the isidentifier() check but fails sqlite3 execution,
    # catching it with OperationalError is fine.
    with pytest.raises(sqlite3.OperationalError, match="no such column: bad_column"):
        db.update_memory("episodic", "some_id", {"bad_column": "value"}, "t1", "p1")

    # Real injection attempt which fails the isidentifier() check
    with pytest.raises(ValueError, match="Invalid column name: .*"):
        db.update_memory("episodic", "some_id", {"; DROP TABLE memories_episodic; --": "value"}, "t1", "p1")

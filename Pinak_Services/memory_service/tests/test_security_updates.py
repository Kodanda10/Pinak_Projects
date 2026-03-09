import pytest
from app.core.database import DatabaseManager

@pytest.fixture
def db(tmp_path):
    return DatabaseManager(str(tmp_path / "test.db"))

def test_database_update_memory_mass_assignment_sql_injection(db):
    res_ep = db.add_episodic("experience", "t1", "p1", goal="be cool", plan=["step1"], tool_logs=[{"tool": "cmd"}])
    mid = res_ep["id"]

    # Valid column name
    success = db.update_memory("episodic", mid, {"goal": "be uncool"}, "t1", "p1")
    assert success is True

    # Invalid column name - integer
    with pytest.raises(ValueError, match="Invalid column name: 1"):
        db.update_memory("episodic", mid, {1: "value"}, "t1", "p1")

    # Invalid column name - SQL injection via string
    with pytest.raises(ValueError, match="Invalid column name: content = \\?, tenant = 'attacker' --"):
        db.update_memory("episodic", mid, {"content = ?, tenant = 'attacker' --": "value"}, "t1", "p1")

    # Verify the value was actually updated to 'be uncool' and not modified by SQL injection
    mem = db.get_memory("episodic", mid, "t1", "p1")
    assert mem["goal"] == "be uncool"
    assert mem["tenant"] == "t1"

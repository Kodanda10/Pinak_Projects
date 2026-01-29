import pytest
from app.core.database import DatabaseManager

@pytest.fixture
def db(tmp_path):
    return DatabaseManager(str(tmp_path / "target.db"))

def test_database_procedural_gap(db):
    # Coverage for 346-349, 360-362
    db.add_procedural("fix logic", ["step1", "step2"], "t1", "p1", trigger="error")
    results = db.search_keyword("fix", "t1", "p1")
    
    found = False
    for r in results:
        if r["type"] == "procedural":
            assert "steps" in r
            assert r["steps"] == ["step1", "step2"]
            found = True
    assert found

def test_database_update_delete_layer_errors(db):
    # Coverage for 275, 283, 303 (if they weren't covered by previous test)
    with pytest.raises(ValueError, match="Invalid layer"):
        db.update_memory("ghost", "id", {}, "t", "p")
    
    with pytest.raises(ValueError, match="Invalid layer"):
        db.delete_memory("ghost", "id", "t", "p")

def test_database_episodic_search_parsing(db):
    db.add_episodic("trip report", "t1", "p1", plan=["go"], tool_logs=[{"t": 1}])
    results = db.search_keyword("trip", "t1", "p1")
    
    found = False
    for r in results:
        if r["type"] == "episodic":
            assert r["plan"] == ["go"]
            assert r["tool_logs"] == [{"t": 1}]
            found = True
    assert found

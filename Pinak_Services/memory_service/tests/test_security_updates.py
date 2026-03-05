import pytest
import os
import json
from app.services.memory_service import MemoryService
from app.core.database import DatabaseManager

@pytest.fixture
def memory_service(tmp_path):
    config = {
        "embedding_model": "dummy",
        "data_root": str(tmp_path / "data"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return MemoryService(config_path=str(config_path))

def test_mass_assignment_protection(memory_service):
    tenant = "t1"
    pid = "p1"

    # Add a memory
    memory = type('obj', (object,), {'content': 'hello world', 'tags': []})
    res = memory_service.add_memory(memory, tenant, pid)
    mid = res.id

    # Verify initial tenant
    item = memory_service.db.get_memory("semantic", mid, tenant, pid)
    assert item['tenant'] == tenant
    assert item['content'] == 'hello world'

    # Attempt mass assignment
    updates = {
        "content": "hacked content",
        "tenant": "hacked_tenant",
        "project_id": "hacked_project",
        "invalid_field": "hacked"
    }

    memory_service.update_memory("semantic", mid, updates, tenant, pid)

    # Verify tenant/project unchanged, but content changed
    item = memory_service.db.get_memory("semantic", mid, tenant, pid)
    assert item is not None
    assert item['tenant'] == tenant
    assert item['project_id'] == pid
    assert item['content'] == 'hacked content'
    assert 'invalid_field' not in item

def test_sql_injection_protection_in_database(memory_service):
    tenant = "t1"
    pid = "p1"

    db = memory_service.db
    res = db.add_semantic("hello", [], tenant, pid, 1)
    mid = res['id']

    # Direct DB layer test for valid identifiers
    with pytest.raises(ValueError) as excinfo:
        db.update_memory("semantic", mid, {"content = 'hacked' --": "value"}, tenant, pid)

    assert "Invalid column name for update" in str(excinfo.value)

    with pytest.raises(ValueError) as excinfo:
        db.update_memory("semantic", mid, {"(SELECT 1)": "value"}, tenant, pid)

    assert "Invalid column name for update" in str(excinfo.value)

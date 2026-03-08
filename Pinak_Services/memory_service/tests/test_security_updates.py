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

def test_sql_injection_prevention(memory_service):
    tenant = "t1"
    pid = "p1"

    # Add a legitimate memory first
    res = memory_service.add_memory(
        type('obj', (object,), {'content': 'original content', 'tags': ['tag1']}),
        tenant, pid
    )
    mid = res.id

    # Try an injection attack in the field key
    injection_key = "content = 'hacked', id"
    updates = {
        injection_key: "'123'",
        "content": "new valid content"
    }

    with pytest.raises(ValueError) as exc:
        memory_service.db.update_memory("semantic", mid, updates, tenant, pid)

    assert "Invalid column name" in str(exc.value)

def test_mass_assignment_prevention(memory_service):
    tenant = "t1"
    pid = "p1"

    # Add a semantic memory
    res = memory_service.add_memory(
        type('obj', (object,), {'content': 'original content', 'tags': ['tag1']}),
        tenant, pid
    )
    mid = res.id

    # Retrieve the original state
    original_item = memory_service.db.get_memory("semantic", mid, tenant, pid)
    assert original_item['tenant'] == tenant
    assert original_item['content'] == "original content"

    # Attempt mass assignment
    malicious_updates = {
        "content": "updated content",
        "tenant": "hacked_tenant", # System field
        "project_id": "hacked_project", # System field
        "embedding_id": 99999, # System field
        "salience": 100, # Field belonging to a different layer (episodic)
        "fake_field": "fake_value" # Non-existent field
    }

    # Perform update through the service layer which now has the whitelist
    success = memory_service.update_memory("semantic", mid, malicious_updates, tenant, pid)
    assert success is True

    # Verify only the allowed fields were updated
    updated_item = memory_service.db.get_memory("semantic", mid, tenant, pid)

    assert updated_item['content'] == "updated content"
    assert updated_item['tenant'] == tenant # Should not change
    assert updated_item['project_id'] == pid # Should not change
    assert updated_item['embedding_id'] == original_item['embedding_id'] # Should not change

    # Ensure no exceptions were thrown by the database for invalid columns
    # and that the layer boundaries are respected.

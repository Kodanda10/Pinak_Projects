import pytest
import os
import json
from app.services.memory_service import MemoryService

@pytest.fixture
def memory_service(tmp_path):
    config = {
        "embedding_model": "dummy",
        "data_root": str(tmp_path / "data"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return MemoryService(config_path=str(config_path))

def test_mass_assignment_prevented(memory_service):
    tenant = "t_secure"
    pid = "p_secure"

    # Add memory
    res = memory_service.add_memory(type('obj', (object,), {'content': 'initial content', 'tags': ['tag1']}), tenant, pid)
    mid = res.id

    # Attempt to update a forbidden field (e.g., client_id, agent_id) and a valid field
    updates = {
        "content": "updated content",
        "client_id": "malicious_client_id",
        "agent_id": "malicious_agent_id",
        "embedding_id": 999999
    }
    success = memory_service.update_memory("semantic", mid, updates, tenant, pid)
    assert success is True

    # Verify only the valid field was updated
    item = memory_service.db.get_memory("semantic", mid, tenant, pid)
    assert item["content"] == "updated content"
    assert item.get("client_id") != "malicious_client_id"
    assert item.get("agent_id") != "malicious_agent_id"

def test_update_only_invalid_fields_fails(memory_service):
    tenant = "t_secure"
    pid = "p_secure"

    # Add memory
    res = memory_service.add_memory(type('obj', (object,), {'content': 'initial content', 'tags': ['tag1']}), tenant, pid)
    mid = res.id

    # Attempt to update only forbidden fields
    updates = {
        "client_id": "malicious_client_id",
        "agent_id": "malicious_agent_id",
        "tenant": "malicious_tenant"
    }

    success = memory_service.update_memory("semantic", mid, updates, tenant, pid)
    assert success is False # Should return False as no valid fields were provided

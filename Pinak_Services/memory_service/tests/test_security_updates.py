import pytest
import os
import json
import sqlite3
from app.services.memory_service import MemoryService
from app.core.schemas import MemoryCreate

@pytest.fixture
def memory_service(tmp_path):
    # Setup config
    data_root = tmp_path / "data"
    data_root.mkdir()
    config = {
        "embedding_model": "dummy",
        "data_root": str(data_root),
    }
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    # Return service
    return MemoryService(config_path=str(config_path))

def test_secure_update_filtering(memory_service):
    """
    Verifies that update_memory filters out keys not in the ALLOWED_UPDATES whitelist,
    preventing SQL injection and schema pollution.
    """
    tenant = "test_tenant"
    project_id = "test_project"

    # Create a memory
    memory_data = MemoryCreate(content="Initial content", tags=["tag1"])
    result = memory_service.add_memory(
        memory_data,
        tenant=tenant,
        project_id=project_id
    )
    memory_id = result.id

    # 1. Attempt to update with ONLY an invalid/malicious key
    # Malicious key example: "content = 'pwned'; DELETE FROM memories_semantic; --"
    # Or just a non-existent column which would normally raise an error
    updates = {"non_existent_column_xyz": "value"}

    # Should return False because all keys are filtered out -> safe_updates is empty
    success = memory_service.update_memory("semantic", memory_id, updates, tenant, project_id)
    assert success is False, "Update should fail when no valid keys are provided"

    # 2. Attempt to update with mixed valid and invalid keys
    updates = {"content": "New Content", "non_existent_column_xyz": "value"}

    # Should succeed for 'content' and silently ignore the invalid key
    success = memory_service.update_memory("semantic", memory_id, updates, tenant, project_id)
    assert success is True, "Update should succeed for valid keys even if invalid ones are present"

    # Verify content changed
    item = memory_service.db.get_memory("semantic", memory_id, tenant, project_id)
    assert item["content"] == "New Content"

    # 3. Verify System Fields cannot be updated even if whitelisted (though they are not)
    # 'id' is in forbidden_keys.
    original_id = item["id"]
    updates = {"id": "new_fake_id", "content": "Another Update"}

    success = memory_service.update_memory("semantic", memory_id, updates, tenant, project_id)
    assert success is True

    item_after = memory_service.db.get_memory("semantic", memory_id, tenant, project_id)
    assert item_after["id"] == original_id
    assert item_after["content"] == "Another Update"

def test_episodic_whitelist(memory_service):
    """Verify episodic layer specific whitelist"""
    tenant = "t1"
    pid = "p1"

    # Create episodic memory directly via DB or service (service add_episodic returns dict)
    res = memory_service.add_episodic("event", tenant, pid)
    mid = res["id"]

    # Allowed: goal
    updates = {"goal": "new goal", "bad_key": "val"}
    success = memory_service.update_memory("episodic", mid, updates, tenant, pid)
    assert success is True

    item = memory_service.db.get_memory("episodic", mid, tenant, pid)
    assert item["goal"] == "new goal"

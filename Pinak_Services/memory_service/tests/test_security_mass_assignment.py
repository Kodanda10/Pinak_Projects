import pytest
import os
import json
from app.services.memory_service import MemoryService
from app.core.schemas import MemoryCreate

@pytest.fixture
def memory_service(tmp_path):
    config = {
        "embedding_model": "dummy",
        "data_root": str(tmp_path / "data"),
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    # Override env vars for consistent testing
    os.environ["PINAK_CONFIG_PATH"] = str(config_path)
    return MemoryService(config_path=str(config_path))

def test_mass_assignment_prevention(memory_service):
    """
    Verify that Mass Assignment vulnerability is fixed by whitelist.
    Only explicitly allowed fields should be updatable.
    """
    tenant = "t-sec"
    project = "p-sec"
    client_id = "client-original"

    # 1. Create a memory
    mem_data = MemoryCreate(content="original content", tags=["secret"])
    result = memory_service.add_memory(
        mem_data,
        tenant,
        project,
        client_id=client_id,
        client_name="Original Client"
    )
    mid = result.id

    # 2. Attempt Mass Assignment with ONLY disallowed fields
    updates_hack = {
        "client_id": "client-hacked",
        "client_name": "Hacked Client"
    }
    success = memory_service.update_memory("semantic", mid, updates_hack, tenant, project)
    assert success is False # Should fail because no fields to update

    # Verify NO change
    item = memory_service.db.get_memory("semantic", mid, tenant, project)
    assert item["client_id"] == "client-original"
    assert item["client_name"] == "Original Client"

    # 3. Attempt Mixed Update (Allowed + Disallowed)
    updates_mixed = {
        "content": "Updated content",   # Allowed
        "client_id": "client-hacked-2" # Disallowed
    }
    success = memory_service.update_memory("semantic", mid, updates_mixed, tenant, project)
    assert success is True # Should succeed for allowed fields

    # Verify PARTIAL update
    updated_item = memory_service.db.get_memory("semantic", mid, tenant, project)
    assert updated_item["content"] == "Updated content"      # Updated
    assert updated_item["client_id"] == "client-original"    # PROTECTED


import pytest
import os
import shutil
import json
from app.services.memory_service import MemoryService
from app.core.schemas import MemoryCreate

# Determine absolute paths based on the test file location
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICE_ROOT = os.path.dirname(TEST_DIR) # Pinak_Services/memory_service
APP_CORE_DIR = os.path.join(SERVICE_ROOT, "app", "core")

@pytest.fixture
def memory_service():
    # Use a unique config for isolation
    config_filename = "config_test_mass_assignment.json"
    config_path = os.path.join(APP_CORE_DIR, config_filename)
    data_root = os.path.join(SERVICE_ROOT, "data_test_mass_assignment")

    if os.path.exists(data_root):
        shutil.rmtree(data_root)

    os.makedirs(data_root, exist_ok=True)

    # Mock config
    with open(config_path, "w") as f:
        json.dump({"data_root": data_root}, f)

    service = MemoryService(config_path=config_path)

    yield service

    # Cleanup
    if os.path.exists(data_root):
        shutil.rmtree(data_root)
    if os.path.exists(config_path):
        os.remove(config_path)

def test_mass_assignment_prevention(memory_service):
    """
    Verify that updating protected fields like 'agent_id' is blocked,
    while legitimate updates like 'content' are allowed.
    """
    tenant = "test-tenant"
    project_id = "test-project"

    # 1. Create a memory
    memory_data = MemoryCreate(content="Initial content", tags=["tag1"])
    result = memory_service.add_memory(
        memory_data,
        tenant=tenant,
        project_id=project_id,
        agent_id="original-agent",
        client_id="original-client",
        client_name="original-client-name"
    )
    memory_id = result.id

    # Verify initial state
    memory = memory_service.db.get_memory("semantic", memory_id, tenant, project_id)
    assert memory["agent_id"] == "original-agent"
    assert memory["client_id"] == "original-client"
    assert memory["content"] == "Initial content"

    # 2. Attempt to update 'agent_id' via mass assignment
    updates = {
        "content": "Updated content",
        "agent_id": "hacked-agent", # This should be ignored
        "client_id": "hacked-client" # This should be ignored
    }

    success = memory_service.update_memory("semantic", memory_id, updates, tenant, project_id)
    assert success is True, "Update should succeed for valid fields"

    # 3. Verify that protected fields were NOT updated, but content WAS
    updated_memory = memory_service.db.get_memory("semantic", memory_id, tenant, project_id)

    assert updated_memory["content"] == "Updated content", "Valid field 'content' should be updated"
    assert updated_memory["agent_id"] == "original-agent", "Protected field 'agent_id' should NOT be updated"
    assert updated_memory["client_id"] == "original-client", "Protected field 'client_id' should NOT be updated"

def test_update_episodic_steps(memory_service):
    """
    Verify that we can update whitelisted 'steps' field in episodic memory.
    """
    tenant = "test-tenant"
    project_id = "test-project"

    # 1. Create episodic memory
    result = memory_service.add_episodic(
        content="Episodic content",
        tenant=tenant,
        project_id=project_id,
        tool_logs=[{"action": "test"}]
    )
    memory_id = result["id"]

    # 2. Update 'steps'
    new_steps = [{"action": "updated"}]
    updates = {"steps": new_steps}

    success = memory_service.update_memory("episodic", memory_id, updates, tenant, project_id)
    assert success is True

    # 3. Verify
    updated_memory = memory_service.db.get_memory("episodic", memory_id, tenant, project_id)
    # The DB stores steps as JSON string, but get_memory deserializes it to list (and copies to tool_logs)
    # Actually DatabaseManager.get_memory:
    # if d.get("steps"): d["steps"] = json.loads(d["steps"]); d["tool_logs"] = d["steps"]

    assert updated_memory["steps"] == new_steps
    assert updated_memory["tool_logs"] == new_steps

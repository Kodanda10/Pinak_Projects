import pytest
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

def test_update_memory_sql_injection_prevention(memory_service):
    tenant = "t1"
    pid = "p1"

    class FakeMemory:
        content = "secret"
        tags = []

    res = memory_service.add_memory(FakeMemory(), tenant, pid)
    mid = res.id

    # Try an injection vector as a dictionary key
    updates = {
        "content = 'hacked' /*": "value"
    }

    with pytest.raises(ValueError, match="Invalid update key"):
        memory_service.update_memory("semantic", mid, updates, tenant, pid)

import pytest
from app.services.memory_service import MemoryService
import json

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
    res = memory_service.add_memory(type('obj', (object,), {'content': 'hello', 'tags': ['a']}), tenant, pid)
    mid = res.id

    with pytest.raises(ValueError, match="Invalid update key"):
        memory_service.update_memory("semantic", mid, {"content = 'test'; DROP TABLE memories_semantic; --": "hello world"}, tenant, pid)

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

def test_update_and_delete(memory_service):
    tenant = "t1"
    pid = "p1"

    # Add
    res = memory_service.add_memory(type('obj', (object,), {'content': 'hello', 'tags': ['a']}), tenant, pid)
    mid = res.id

    # Update
    memory_service.update_memory("semantic", mid, {"content": "hello world"}, tenant, pid)

    # Verify Update (and re-embedding logic implicitly via search, but lets check DB content)
    item = memory_service.db.get_memory("semantic", mid, tenant, pid)
    assert item['content'] == "hello world"

    # Delete
    memory_service.delete_memory("semantic", mid, tenant, pid)
    item = memory_service.db.get_memory("semantic", mid, tenant, pid)
    assert item is None

def test_json_deserialization(memory_service):
    tenant = "t2"
    pid = "p2"

    # Add with tags
    res = memory_service.add_memory(type('obj', (object,), {'content': 'json test', 'tags': ['tag1', 'tag2']}), tenant, pid)
    mid = res.id

    # Get direct
    item = memory_service.db.get_memory("semantic", mid, tenant, pid)
    assert isinstance(item['tags'], list)
    assert 'tag1' in item['tags']

    # Search
    results = memory_service.search_hybrid("json", tenant, pid)
    assert len(results) > 0
    assert isinstance(results[0]['tags'], list)

def test_update_security_mass_assignment(memory_service):
    """Verify that unauthorized fields are ignored during update."""
    tenant = "t3"
    pid = "p3"

    # Add
    res = memory_service.add_memory(
        type('obj', (object,), {'content': 'safe', 'tags': []}),
        tenant, pid, client_id="safe_client"
    )
    mid = res.id

    # Attempt to update client_id (should be ignored)
    updates = {
        "content": "still safe",
        "client_id": "hacked_client"
    }
    memory_service.update_memory("semantic", mid, updates, tenant, pid)

    item = memory_service.db.get_memory("semantic", mid, tenant, pid)
    assert item['content'] == "still safe"
    assert item['client_id'] == "safe_client"  # Must NOT be "hacked_client"

def test_update_security_sql_injection_attempt(memory_service):
    """Verify that keys with potential SQL injection are ignored."""
    tenant = "t4"
    pid = "p4"

    res = memory_service.add_memory(
        type('obj', (object,), {'content': 'original', 'tags': []}),
        tenant, pid
    )
    mid = res.id

    # Attempt injection via key
    updates = {
        "content = 'pwned', client_id": "malicious"
    }
    # This key is not in ALLOWED_UPDATES, so it should be filtered out
    memory_service.update_memory("semantic", mid, updates, tenant, pid)

    item = memory_service.db.get_memory("semantic", mid, tenant, pid)
    assert item['content'] == "original"

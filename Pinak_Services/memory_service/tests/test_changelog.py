import os
import json
from app.services.memory_service import add_changelog, list_changelog

def test_add_and_list_changelog_service(memory_service):
    """Test adding and listing changelog entries at the service level."""
    tenant = "test_tenant"
    project_id = "test_project"
    entity_id = "test_entity"
    layer = "test_layer"
    old_value = {"field": "old"}
    new_value = {"field": "new"}

    # Add a changelog entry
    added_entry = add_changelog(memory_service, tenant, project_id, entity_id, layer, old_value, new_value)
    assert added_entry['entity_id'] == entity_id
    assert added_entry['layer'] == layer
    assert added_entry['old_value'] == old_value
    assert added_entry['new_value'] == new_value

    # List changelog entries
    changelog_entries = list_changelog(memory_service, tenant, project_id)
    assert len(changelog_entries) > 0
    assert any(entry['entity_id'] == entity_id for entry in changelog_entries)

def test_changelog_api_endpoints(client):
    """Test the /changelog/add and /changelog/list API endpoints."""
    project_id = "test_project_api"
    headers = {"X-Pinak-Project": project_id}

    # Add a changelog entry via the API
    add_payload = {
        "entity_id": "api_entity",
        "layer": "api_layer",
        "old_value": {"api_field": "old"},
        "new_value": {"api_field": "new"}
    }
    response = client.post("/api/v1/memory/changelog/add", headers=headers, json=add_payload)
    assert response.status_code == 201
    added_entry = response.json()
    assert added_entry['entity_id'] == "api_entity"

    # List changelog entries via the API
    response = client.get("/api/v1/memory/changelog/list", headers=headers)
    assert response.status_code == 200
    changelog_entries = response.json()
    assert len(changelog_entries) > 0
    assert any(entry['entity_id'] == "api_entity" for entry in changelog_entries)

def test_read_jsonl_utility_functions(memory_service):
    """Test the _read_jsonl_file and _read_jsonl_from_directory utility functions."""
    tenant = "test_tenant_jsonl"
    project_id = "test_project_jsonl"
    base_dir = memory_service._store_dir(tenant, project_id)
    changelog_dir = os.path.join(base_dir, 'changelog')
    os.makedirs(changelog_dir, exist_ok=True)

    # Create a dummy JSONL file
    file_path = os.path.join(changelog_dir, "changelog_20250101.jsonl")
    records = [
        {"id": 1, "data": "record1"},
        {"id": 2, "data": "record2"}
    ]
    with open(file_path, 'w') as f:
        for record in records:
            f.write(json.dumps(record) + '\n')

    # Test _read_jsonl_file
    read_records = memory_service._read_jsonl_file(file_path)
    assert read_records == records

    # Test _read_jsonl_from_directory
    read_records_dir = memory_service._read_jsonl_from_directory(changelog_dir, 'changelog')
    assert read_records_dir == records
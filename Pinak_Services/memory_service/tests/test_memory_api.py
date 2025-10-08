from app.core.schemas import MemoryCreate

def test_add_memory(client):
    response = client.post("/api/v1/memory/add", json={"content": "test memory", "tags": ["test"]})
    assert response.status_code == 201
    assert "id" in response.json()
    assert response.json()["content"] == "test memory"

def test_retrieve_memory(client):
    memory_content = "The sky is blue on a clear day."
    add_response = client.post("/api/v1/memory/add", json={"content": memory_content, "tags": ["sky"]})
    assert add_response.status_code == 201

    search_response = client.get(f"/api/v1/memory/search?query=sky")
    assert search_response.status_code == 200
    response_data = search_response.json()
    assert isinstance(response_data, list)
    assert len(response_data) > 0
    assert response_data[0]['content'] == memory_content

def test_add_episodic_memory(client):
    project_id = "episodic_project"
    headers = {"X-Pinak-Project": project_id}
    response = client.post("/api/v1/memory/episodic/add", headers=headers, json={"content": "episodic test", "salience": 5})
    assert response.status_code == 201
    data = response.json()
    assert "content" in data
    assert data["salience"] == 5

def test_list_episodic_memory(client):
    project_id = "list_episodic_project"
    headers = {"X-Pinak-Project": project_id}
    client.post("/api/v1/memory/episodic/add", headers=headers, json={"content": "episodic 1", "salience": 3})
    client.post("/api/v1/memory/episodic/add", headers=headers, json={"content": "episodic 2", "salience": 7})

    response = client.get("/api/v1/memory/episodic/list", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2

def test_full_memory_workflow(client):
    """Test complete memory workflow across layers."""
    project_id = "full_workflow_project"
    headers = {"X-Pinak-Project": project_id}

    # 1. Add semantic memory
    client.post("/api/v1/memory/add", json={"content": "Python is great", "tags": ["python"]})
    
    # 2. Add episodic memory
    client.post("/api/v1/memory/episodic/add", headers=headers, json={"content": "Learned Python today", "salience": 9})
    
    # 3. Add procedural memory
    client.post("/api/v1/memory/procedural/add", headers=headers, json={"skill_id": "python_debug", "steps": ["print", "debug"]})
    
    # 4. Add event
    client.post("/api/v1/memory/event", headers=headers, json={"type": "learning", "topic": "python"})
    
    # 5. Search across layers
    response = client.get("/api/v1/memory/search_v2?query=python&layers=episodic,procedural", headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert len(data.get("episodic", [])) > 0
    assert len(data.get("procedural", [])) > 0
    assert "Python" in data["episodic"][0]["content"]
    assert "python" in data["procedural"][0]["skill_id"]
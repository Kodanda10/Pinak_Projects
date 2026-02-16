import os
import jwt
import time
import requests
import json
import datetime

BASE_URL = "http://localhost:8000/api/v1/memory"
SECRET = os.getenv("PINAK_JWT_SECRET", "test_secret")

def mint_token():
    payload = {
        "sub": "test-user",
        "tenant": "default",
        "project_id": "default",
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=10),
        "scopes": ["memory.write", "memory.read", "memory.admin"],
        "roles": ["admin"]
    }
    return jwt.encode(payload, SECRET, algorithm="HS256")

def main():
    token = mint_token()
    headers = {"Authorization": f"Bearer {token}"}

    print(f"Token: {token[:10]}...")

    # 1. Health Check
    print("Checking Health...")
    try:
        resp = requests.get("http://localhost:8000/api/v1/health")
        if resp.status_code != 200:
            print(f"Health check failed: {resp.text}")
            return
        print("Health OK.")
    except Exception as e:
        print(f"Server not reachable: {e}")
        return

    # 2. Register Client
    print("Registering Client...")
    client_data = {
        "client_id": "test-client-1",
        "client_name": "Test Script",
        "status": "trusted"
    }
    resp = requests.post(f"{BASE_URL}/client/register", json=client_data, headers=headers)
    if resp.status_code not in [200, 201]:
        # If client exists it might be 400 or 409, need to handle or assume OK if created before
        if "already exists" in resp.text:
            print("Client already exists.")
        else:
            print(f"Failed to register client: {resp.status_code} {resp.text}")
            return
    else:
        print("Client Registered.")

    # 3. Add Memory
    print("Adding Memory...")
    memory_data = {
        "content": "The user prefers dark mode for all interfaces.",
        "tags": ["preference", "ui"]
    }
    # Need client headers
    mem_headers = headers.copy()
    mem_headers.update({
        "X-Pinak-Client-Id": "test-client-1",
        "X-Pinak-Client-Name": "Test Script"
    })

    resp = requests.post(f"{BASE_URL}/add", json=memory_data, headers=mem_headers)
    if resp.status_code != 201:
        print(f"Failed to add memory: {resp.status_code} {resp.text}")
        return
    print(f"Memory Added: {resp.json().get('id')}")

    # 4. Search Memory
    print("Searching Memory...")
    # Wait a moment for async indexing if any (though vector store is sync in this codebase)
    time.sleep(2)

    query = "dark mode preference"

    # Try Legacy Search First
    print("Trying Legacy Search...")
    resp_legacy = requests.get(f"{BASE_URL}/search", params={"query": query, "k": 5}, headers=mem_headers)
    if resp_legacy.status_code == 200:
        print(f"Legacy Search Results: {json.dumps(resp_legacy.json(), indent=2)}")
    else:
        print(f"Legacy Search Failed: {resp_legacy.status_code} {resp_legacy.text}")

    # Try Unified Search
    print("Trying Unified Search...")
    resp = requests.get(f"{BASE_URL}/retrieve_context", params={"query": query}, headers=mem_headers)

    if resp.status_code != 200:
        print(f"Search failed: {resp.status_code} {resp.text}")
        return

    data = resp.json()
    # Check if we found it
    found = False
    # Data structure: {"context_by_layer": {"semantic": [...]}} (or flattened in some versions, let's check keys)
    print(f"Unified Response Keys: {data.keys()}")

    # Adapt to potential schema
    semantic = data.get("context_by_layer", {}).get("semantic", [])
    # Fallback if structure is different
    if not semantic and "semantic" in data:
        semantic = data["semantic"]

    for item in semantic:
        if "dark mode" in item.get("content", "").lower():
            found = True
            print(f"Found Memory: {item['content']}")
            break

    if found:
        print("✅ SUCCESS: Memory flow Verified.")
    else:
        print("❌ FAILURE: Memory not found in search results.")
        print(f"Results: {json.dumps(data, indent=2)}")

if __name__ == "__main__":
    main()

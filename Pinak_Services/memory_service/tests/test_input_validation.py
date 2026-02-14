import datetime
import jwt
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture(autouse=True)
def _set_secret(monkeypatch):
    monkeypatch.setenv("PINAK_JWT_SECRET", "test-secret")

def _issue_token():
    payload = {
        "sub": "tester",
        "tenant": "t1",
        "project_id": "p1",
        "roles": ["user"],
        "scopes": ["memory.write", "memory.read"],
        "iat": datetime.datetime.now(datetime.UTC),
        "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=5),
    }
    return jwt.encode(payload, "test-secret", algorithm="HS256")

def test_add_semantic_empty_content_should_fail():
    token = _issue_token()
    headers = {"Authorization": f"Bearer {token}"}
    client = TestClient(app)

    # Attempt to add empty content (violates minLength: 1)
    resp = client.post(
        "/api/v1/memory/add",
        json={"content": "", "tags": ["test"]},
        headers=headers
    )

    assert resp.status_code == 422
    assert "Schema validation failed" in resp.json()["detail"]["message"]

def test_add_semantic_huge_content_should_fail():
    token = _issue_token()
    headers = {"Authorization": f"Bearer {token}"}
    client = TestClient(app)

    # 2MB string (violates maxLength: 65536)
    huge_content = "a" * (2 * 1024 * 1024)

    resp = client.post(
        "/api/v1/memory/add",
        json={"content": huge_content, "tags": ["huge"]},
        headers=headers
    )

    assert resp.status_code == 422
    assert "Schema validation failed" in resp.json()["detail"]["message"]

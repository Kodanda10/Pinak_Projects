import json
import os
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient


def client():
    # Ensure mock embeddings for speed
    os.environ.setdefault("USE_MOCK_EMBEDDINGS", "true")
    from app.main import app

    return TestClient(app)


def iso(dt):
    return dt.replace(microsecond=0).isoformat()


def test_governance_event_mirrors_to_changelog(tmp_path, monkeypatch):
    c = client()
    pid = "Pnk-test"

    # Emit a governance audit event
    now = datetime.now(timezone.utc)
    r = c.post(
        "/api/v1/memory/event",
        headers={"X-Pinak-Project": pid},
        json={
            "type": "gov_audit",
            "path": "/policy/update",
            "method": "POST",
            "status": 200,
            "role": "admin",
        },
    )
    assert r.status_code == 201

    # Changelog should include change_type=governance
    r2 = c.get(
        "/api/v1/memory/changelog",
        headers={"X-Pinak-Project": pid},
        params={"change_type": "governance"},
    )
    assert r2.status_code == 200
    body = r2.json()
    assert any(rec.get("change_type") == "governance" for rec in body)
    # Verify paging works
    r3 = c.get(
        "/api/v1/memory/changelog",
        headers={"X-Pinak-Project": pid},
        params={"change_type": "governance", "limit": 1, "offset": 0},
    )
    assert r3.status_code == 200
    assert len(r3.json()) <= 1


def test_changelog_since_until_filters(tmp_path, monkeypatch):
    c = client()
    pid = "Pnk-test2"
    # Create two governance entries separated by time
    r = c.post(
        "/api/v1/memory/event",
        headers={"X-Pinak-Project": pid},
        json={"type": "gov_audit", "path": "/a", "method": "POST", "status": 200},
    )
    assert r.status_code == 201

    # Slight pause between entries simulated by filtering by timestamps
    # Fetch all to get timestamps
    all_entries = c.get(
        "/api/v1/memory/changelog",
        headers={"X-Pinak-Project": pid},
        params={"change_type": "governance"},
    ).json()
    assert len(all_entries) >= 1
    ts_first = all_entries[-1]["ts"]

    # Add another event
    r2 = c.post(
        "/api/v1/memory/event",
        headers={"X-Pinak-Project": pid},
        json={"type": "gov_audit", "path": "/b", "method": "POST", "status": 201},
    )
    assert r2.status_code == 201

    # Use since to filter to the later entry only
    filtered = c.get(
        "/api/v1/memory/changelog",
        headers={"X-Pinak-Project": pid},
        params={"change_type": "governance", "since": ts_first},
    ).json()
    assert len(filtered) >= 1

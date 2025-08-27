import os
import json
import pytest
from fastapi.testclient import TestClient
from jose import jwt


def make_client():
    os.environ.setdefault('GOV_UPSTREAM', 'http://example-upstream')  # won't be called in tests
    os.environ.setdefault('SECRET_KEY', 'test-secret')
    os.environ.setdefault('REQUIRE_PROJECT_HEADER', 'true')
    os.environ.setdefault('PINAK_ALLOWED_ROLES', 'viewer,editor,admin')
    from app.main import app
    return TestClient(app)


def mint(pid: str, role: str | None = None, secret: str = 'test-secret') -> str:
    claims = {'pid': pid, 'sub': 'tester'}
    if role:
        claims['role'] = role
    return jwt.encode(claims, secret, algorithm='HS256')


def test_missing_project_header_returns_400():
    c = make_client()
    r = c.get('/any/path')
    assert r.status_code == 400


def test_pid_header_token_mismatch_returns_403():
    c = make_client()
    tok = mint('Pnk-a')
    r = c.get('/any/path', headers={'X-Pinak-Project': 'Pnk-b', 'Authorization': f'Bearer {tok}'})
    assert r.status_code == 403


def test_role_allowlist_denies_unlisted_role():
    os.environ['PINAK_ALLOWED_ROLES'] = 'viewer'  # tighten allowlist
    c = make_client()
    tok = mint('Pnk-a', role='admin')
    r = c.get('/any/path', headers={'X-Pinak-Project': 'Pnk-a', 'Authorization': f'Bearer {tok}'})
    assert r.status_code == 403


def test_role_forwarding_header_present_on_proxy(monkeypatch):
    # Intercept httpx client to verify forwarded headers without real upstream call
    import httpx
    captured = {}

    async def fake_request(self, method, url, headers=None, content=None):  # noqa: ANN001
        captured['headers'] = headers or {}
        # minimal httpx.Response mock
        return httpx.Response(200, json={'ok': True})

    monkeypatch.setattr(httpx.AsyncClient, 'request', fake_request, raising=True)
    c = make_client()
    tok = mint('Pnk-a', role='editor')
    r = c.get('/any/path', headers={'X-Pinak-Project': 'Pnk-a', 'Authorization': f'Bearer {tok}'})
    assert r.status_code == 200
    # Role should be forwarded
    assert captured['headers'].get('X-Pinak-Role') == 'editor'


def test_audit_sync_posts_to_memory_api(monkeypatch):
    # Verify that for mutating requests the gateway posts an audit event to memory API
    import httpx
    captured = {'upstream': None, 'audit': None}
    os.environ['MEMORY_API_URL'] = 'http://memory-api:8000'

    async def fake_request(self, method, url, headers=None, content=None):  # noqa: ANN001
        # First call is to upstream Parlant; capture it and return success
        if captured['upstream'] is None:
            captured['upstream'] = {'method': method, 'url': url}
            return httpx.Response(200, json={'ok': True})
        # Second call is the audit POST to memory API
        captured['audit'] = {'method': method, 'url': url, 'headers': headers, 'content': content}
        return httpx.Response(200, json={'status': 'ok'})

    monkeypatch.setattr(httpx.AsyncClient, 'request', fake_request, raising=True)
    c = make_client()
    tok = mint('Pnk-a', role='admin')
    r = c.post('/policy/update', headers={'X-Pinak-Project': 'Pnk-a', 'Authorization': f'Bearer {tok}'}, json={'x': 1})
    assert r.status_code == 200
    assert captured['upstream'] is not None
    assert captured['audit'] is not None
    # Ensure the audit call targets memory API event endpoint
    assert '/api/v1/memory/event' in captured['audit']['url']

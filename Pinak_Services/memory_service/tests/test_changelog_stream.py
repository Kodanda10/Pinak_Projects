import os, pytest, json
from httpx import AsyncClient, ASGITransport
from asgi_lifespan import LifespanManager
from app.main import app

os.environ.setdefault('USE_MOCK_EMBEDDINGS','true')

@pytest.mark.asyncio
async def test_changelog_create_and_redact():
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as ac:
            # add creates a changelog entry (best-effort)
            r = await ac.post('/api/v1/memory/add', json={'content':'to redact','tags':[]})
            assert r.status_code==201
            mid = r.json()['id']
            # redact
            rr = await ac.post('/api/v1/memory/changelog/redact', json={'memory_id': mid, 'reason':'cleanup'})
            assert rr.status_code==200
            # list changelog
            cl = await ac.get('/api/v1/memory/changelog')
            assert cl.status_code==200
            body = cl.json()
            assert any(e.get('change_type')=='create' for e in body)
            assert any(e.get('change_type')=='redact' and e.get('target_id')==mid for e in body)

import os, pytest
from httpx import AsyncClient, ASGITransport
from asgi_lifespan import LifespanManager
from jose import jwt

os.environ.setdefault('USE_MOCK_EMBEDDINGS','true')
os.environ.setdefault('REQUIRE_PROJECT_HEADER','true')
SECRET='change-me-in-prod'

from app.main import app

def mint(pid: str):
    return jwt.encode({'sub':'t','pid':pid}, SECRET, algorithm='HS256')

@pytest.mark.asyncio
async def test_pid_must_match_header():
    pid='Pnk-A'
    tok = mint(pid)
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as ac:
            r_ok = await ac.post('/api/v1/memory/episodic/add', headers={'Authorization':f'Bearer {tok}','X-Pinak-Project':pid}, json={'content':'x'})
            assert r_ok.status_code==201
            r_bad = await ac.post('/api/v1/memory/episodic/add', headers={'Authorization':f'Bearer {tok}','X-Pinak-Project':'Pnk-B'}, json={'content':'x'})
            assert r_bad.status_code==403


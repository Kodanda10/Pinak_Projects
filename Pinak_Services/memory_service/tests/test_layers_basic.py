
import pytest
from httpx import AsyncClient, ASGITransport
from asgi_lifespan import LifespanManager
from app.main import app

@pytest.mark.asyncio
async def test_layers_endpoints_basic():
    pid='Pnk-L'
    async with LifespanManager(app):
        async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as ac:
            r1 = await ac.post('/api/v1/memory/episodic/add', headers={'X-Pinak-Project':pid}, json={'content':'ep mem','salience':5})
            assert r1.status_code==201
            r2 = await ac.post('/api/v1/memory/procedural/add', headers={'X-Pinak-Project':pid}, json={'skill_id':'build','steps':['a','b']})
            assert r2.status_code==201
            r3 = await ac.post('/api/v1/memory/rag/add', headers={'X-Pinak-Project':pid}, json={'query':'faq','external_source':'kb'})
            assert r3.status_code==201
            sv2 = await ac.get('/api/v1/memory/search_v2', headers={'X-Pinak-Project':pid}, params={'query':'ep','layers':'semantic,episodic,procedural,rag'})
            assert sv2.status_code==200
            data = sv2.json()
            assert 'episodic' in data and any('ep' in i.get('content','') for i in data['episodic'])

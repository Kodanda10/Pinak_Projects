"""
Session Memory Tests - Service Level
Tests session memory functionality with TTL support
"""

import os
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
import json

os.environ.setdefault('USE_MOCK_EMBEDDINGS', 'true')

from app.services.memory_service import MemoryService


class TestSessionMemory:
    """Test session memory functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data"""
        temp_path = tempfile.mkdtemp(prefix="test_session_memory_")
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def memory_service(self, temp_dir):
        """Create MemoryService for testing"""
        return MemoryService(data_dir=temp_dir)

    def test_add_session_memory(self, memory_service):
        """Test adding session memory"""
        tenant = "test_tenant"
        project_id = "test_project"
        payload = {
            "session_id": "session_123",
            "content": "User session data",
            "ttl_seconds": 3600
        }

        # Simulate the session_add endpoint logic
        base = memory_service._store_dir(tenant, project_id)
        sid = payload.get('session_id') or 'default'
        path = memory_service._session_file(base, sid)

        rec = {
            'session_id': sid,
            'content': payload.get('content') or '',
            'project_id': project_id,
            'ts': datetime.utcnow().isoformat(),
        }

        ttl = payload.get('ttl_seconds')
        if ttl:
            rec['expires_at'] = (datetime.utcnow() + timedelta(seconds=int(ttl))).isoformat()

        memory_service._append_jsonl(path, rec)

        # Verify the file was created and contains the data
        assert os.path.exists(path)

        with open(path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        assert len(data) == 1
        assert data[0]['session_id'] == sid
        assert data[0]['content'] == payload['content']
        assert 'expires_at' in data[0]

    def test_list_session_memories(self, memory_service):
        """Test listing session memories"""
        tenant = "test_tenant"
        project_id = "test_project"
        session_id = "test_session"

        # Add multiple session records
        base = memory_service._store_dir(tenant, project_id)
        path = memory_service._session_file(base, session_id)

        records = [
            {
                'session_id': session_id,
                'content': 'Session data 1',
                'project_id': project_id,
                'ts': datetime.utcnow().isoformat(),
            },
            {
                'session_id': session_id,
                'content': 'Session data 2',
                'project_id': project_id,
                'ts': (datetime.utcnow() + timedelta(minutes=1)).isoformat(),
            }
        ]

        for rec in records:
            memory_service._append_jsonl(path, rec)

        # Simulate list_session logic
        out = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        out.append(obj)
                    except Exception:
                        pass

        assert len(out) == 2
        assert all(item['session_id'] == session_id for item in out)
        assert out[0]['content'] == 'Session data 1'
        assert out[1]['content'] == 'Session data 2'

    def test_session_ttl_expiration(self, memory_service):
        """Test TTL expiration for session memories"""
        tenant = "test_tenant"
        project_id = "test_project"
        session_id = "test_session"

        base = memory_service._store_dir(tenant, project_id)
        path = memory_service._session_file(base, session_id)

        # Add records with different expiration times
        now = datetime.utcnow()
        records = [
            {
                'session_id': session_id,
                'content': 'Active session',
                'project_id': project_id,
                'ts': now.isoformat(),
                'expires_at': (now + timedelta(hours=1)).isoformat()
            },
            {
                'session_id': session_id,
                'content': 'Expired session',
                'project_id': project_id,
                'ts': (now - timedelta(hours=2)).isoformat(),
                'expires_at': (now - timedelta(hours=1)).isoformat()  # Already expired
            }
        ]

        for rec in records:
            memory_service._append_jsonl(path, rec)

        # Simulate list_session with TTL filtering
        out = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        # Filter expired
                        exp = obj.get('expires_at')
                        if exp:
                            try:
                                if datetime.fromisoformat(exp) < datetime.utcnow():
                                    continue
                            except Exception:
                                pass
                        out.append(obj)
                    except Exception:
                        pass

        # Should only return the active session
        assert len(out) == 1
        assert out[0]['content'] == 'Active session'

    def test_session_with_filters(self, memory_service):
        """Test session listing with time filters"""
        tenant = "test_tenant"
        project_id = "test_project"
        session_id = "test_session"

        base = memory_service._store_dir(tenant, project_id)
        path = memory_service._session_file(base, session_id)

        # Add records with different timestamps
        base_time = datetime.utcnow()
        records = [
            {
                'session_id': session_id,
                'content': 'Old session',
                'project_id': project_id,
                'ts': (base_time - timedelta(hours=2)).isoformat(),
            },
            {
                'session_id': session_id,
                'content': 'Recent session',
                'project_id': project_id,
                'ts': base_time.isoformat(),
            }
        ]

        for rec in records:
            memory_service._append_jsonl(path, rec)

        # Test time filtering
        since = (base_time - timedelta(hours=1)).isoformat()
        out = []

        def parse_ts(ts: str):
            try:
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            except Exception:
                return None

        t_since = parse_ts(since) if since else None

        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        ts = parse_ts(obj.get('ts', ''))
                        if t_since and ts and ts < t_since:
                            continue
                        out.append(obj)
                    except Exception:
                        pass

        # Should only return the recent session
        assert len(out) == 1
        assert out[0]['content'] == 'Recent session'


class TestWorkingMemory:
    """Test working memory functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data"""
        temp_path = tempfile.mkdtemp(prefix="test_working_memory_")
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def memory_service(self, temp_dir):
        """Create MemoryService for testing"""
        return MemoryService(data_dir=temp_dir)

    def test_add_working_memory(self, memory_service):
        """Test adding working memory"""
        tenant = "test_tenant"
        project_id = "test_project"
        payload = {
            "content": "Working memory content",
            "ttl_seconds": 1800
        }

        # Simulate working_add endpoint logic
        base = memory_service._store_dir(tenant, project_id)
        path = memory_service._working_file(base)

        rec = {
            'content': payload.get('content') or '',
            'project_id': project_id,
            'ts': datetime.utcnow().isoformat(),
        }

        ttl = payload.get('ttl_seconds')
        if ttl:
            rec['expires_at'] = (datetime.utcnow() + timedelta(seconds=int(ttl))).isoformat()

        memory_service._append_jsonl(path, rec)

        # Verify the data was stored
        assert os.path.exists(path)

        with open(path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]

        assert len(data) == 1
        assert data[0]['content'] == payload['content']
        assert 'expires_at' in data[0]

    def test_list_working_memories(self, memory_service):
        """Test listing working memories"""
        tenant = "test_tenant"
        project_id = "test_project"

        base = memory_service._store_dir(tenant, project_id)
        path = memory_service._working_file(base)

        # Add multiple working memory records
        records = [
            {'content': 'Working 1', 'project_id': project_id, 'ts': datetime.utcnow().isoformat()},
            {'content': 'Working 2', 'project_id': project_id, 'ts': (datetime.utcnow() + timedelta(minutes=1)).isoformat()},
        ]

        for rec in records:
            memory_service._append_jsonl(path, rec)

        # Simulate list_working logic
        out = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        out.append(obj)
                    except Exception:
                        pass

        assert len(out) == 2
        assert out[0]['content'] == 'Working 1'
        assert out[1]['content'] == 'Working 2'

    def test_working_memory_ttl_expiration(self, memory_service):
        """Test TTL expiration for working memories"""
        tenant = "test_tenant"
        project_id = "test_project"

        base = memory_service._store_dir(tenant, project_id)
        path = memory_service._working_file(base)

        now = datetime.utcnow()
        records = [
            {
                'content': 'Active working memory',
                'project_id': project_id,
                'ts': now.isoformat(),
                'expires_at': (now + timedelta(hours=1)).isoformat()
            },
            {
                'content': 'Expired working memory',
                'project_id': project_id,
                'ts': (now - timedelta(hours=2)).isoformat(),
                'expires_at': (now - timedelta(hours=1)).isoformat()
            }
        ]

        for rec in records:
            memory_service._append_jsonl(path, rec)

        # Simulate list_working with TTL filtering
        out = []
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        exp = obj.get('expires_at')
                        if exp:
                            try:
                                if datetime.fromisoformat(exp) < datetime.utcnow():
                                    continue
                            except Exception:
                                pass
                        out.append(obj)
                    except Exception:
                        pass

        assert len(out) == 1
        assert out[0]['content'] == 'Active working memory'

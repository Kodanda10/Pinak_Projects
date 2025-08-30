"""
Event Logging and Audit Tests - Service Level
Tests event logging, audit trails, and changelog functionality
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta

import pytest

os.environ.setdefault("USE_MOCK_EMBEDDINGS", "true")

from app.services.memory_service import MemoryService


class TestEventLogging:
    """Test event logging functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data"""
        temp_path = tempfile.mkdtemp(prefix="test_event_logging_")
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def memory_service(self, temp_dir):
        """Create MemoryService for testing"""
        return MemoryService(data_dir=temp_dir)

    def test_log_memory_event(self, memory_service):
        """Test logging memory events"""
        tenant = "test_tenant"
        project_id = "test_project"
        event_data = {
            "event_type": "memory_add",
            "memory_type": "episodic",
            "content": "Test memory content",
            "metadata": {"source": "test"},
        }

        # Simulate event logging logic
        base = memory_service._store_dir(tenant, project_id)
        path = memory_service._dated_file(base, "events", "events")

        rec = {
            "event_type": event_data.get("event_type"),
            "memory_type": event_data.get("memory_type"),
            "content": event_data.get("content"),
            "metadata": event_data.get("metadata", {}),
            "project_id": project_id,
            "ts": datetime.utcnow().isoformat(),
        }

        memory_service._append_jsonl(path, rec)

        # Verify event was logged
        assert os.path.exists(path)

        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        assert len(data) == 1
        assert data[0]["event_type"] == "memory_add"
        assert data[0]["memory_type"] == "episodic"
        assert data[0]["content"] == "Test memory content"
        assert data[0]["metadata"]["source"] == "test"

    def test_list_events(self, memory_service):
        """Test listing events"""
        tenant = "test_tenant"
        project_id = "test_project"

        base = memory_service._store_dir(tenant, project_id)
        path = memory_service._dated_file(base, "events", "events")

        # Add multiple events
        events = [
            {
                "event_type": "memory_add",
                "memory_type": "episodic",
                "content": "Added episodic memory",
                "project_id": project_id,
                "ts": datetime.utcnow().isoformat(),
            },
            {
                "event_type": "memory_search",
                "memory_type": "rag",
                "content": "Searched RAG memory",
                "project_id": project_id,
                "ts": (datetime.utcnow() + timedelta(minutes=1)).isoformat(),
            },
        ]

        for event in events:
            memory_service._append_jsonl(path, event)

        # Simulate list_events logic
        out = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        out.append(obj)
                    except Exception:
                        pass

        assert len(out) == 2
        assert out[0]["event_type"] == "memory_add"
        assert out[1]["event_type"] == "memory_search"

    def test_events_with_filters(self, memory_service):
        """Test event listing with filters"""
        tenant = "test_tenant"
        project_id = "test_project"

        base = memory_service._store_dir(tenant, project_id)
        path = memory_service._dated_file(base, "events", "events")

        base_time = datetime.utcnow()
        events = [
            {
                "event_type": "memory_add",
                "memory_type": "episodic",
                "content": "Old event",
                "project_id": project_id,
                "ts": (base_time - timedelta(hours=2)).isoformat(),
            },
            {
                "event_type": "memory_search",
                "memory_type": "rag",
                "content": "Recent event",
                "project_id": project_id,
                "ts": base_time.isoformat(),
            },
        ]

        for event in events:
            memory_service._append_jsonl(path, event)

        # Test time filtering
        since = (base_time - timedelta(hours=1)).isoformat()
        out = []

        def parse_ts(ts: str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except Exception:
                return None

        t_since = parse_ts(since) if since else None

        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                        ts = parse_ts(obj.get("ts", ""))
                        if t_since and ts and ts < t_since:
                            continue
                        out.append(obj)
                    except Exception:
                        pass

        assert len(out) == 1
        assert out[0]["content"] == "Recent event"


class TestAuditTrail:
    """Test audit trail functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data"""
        temp_path = tempfile.mkdtemp(prefix="test_audit_trail_")
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def memory_service(self, temp_dir):
        """Create MemoryService for testing"""
        return MemoryService(data_dir=temp_dir)

    def test_audit_memory_operations(self, memory_service):
        """Test auditing memory operations"""
        tenant = "test_tenant"
        project_id = "test_project"

        # Simulate audit trail for memory operations
        base = memory_service._store_dir(tenant, project_id)
        audit_path = memory_service._dated_file(base, "audit", "audit")

        # Add audit records for different operations
        operations = [
            {
                "operation": "add_episodic",
                "memory_id": "mem_123",
                "content": "Episodic memory content",
                "user_id": "user_456",
                "timestamp": datetime.utcnow().isoformat(),
                "project_id": project_id,
            },
            {
                "operation": "search_rag",
                "query": "test query",
                "results_count": 5,
                "user_id": "user_456",
                "timestamp": (datetime.utcnow() + timedelta(minutes=1)).isoformat(),
                "project_id": project_id,
            },
        ]

        for op in operations:
            memory_service._append_jsonl(audit_path, op)

        # Verify audit trail
        assert os.path.exists(audit_path)

        with open(audit_path, "r", encoding="utf-8") as f:
            audit_data = [json.loads(line) for line in f]

        assert len(audit_data) == 2
        assert audit_data[0]["operation"] == "add_episodic"
        assert audit_data[0]["memory_id"] == "mem_123"
        assert audit_data[1]["operation"] == "search_rag"
        assert audit_data[1]["results_count"] == 5

    def test_audit_with_metadata(self, memory_service):
        """Test audit records with metadata"""
        tenant = "test_tenant"
        project_id = "test_project"

        base = memory_service._store_dir(tenant, project_id)
        audit_path = memory_service._dated_file(base, "audit", "audit")

        audit_record = {
            "operation": "memory_update",
            "memory_id": "mem_789",
            "old_content": "Old content",
            "new_content": "New content",
            "user_id": "user_456",
            "ip_address": "192.168.1.100",
            "user_agent": "TestAgent/1.0",
            "metadata": {"reason": "Content correction", "approved_by": "admin_user"},
            "timestamp": datetime.utcnow().isoformat(),
            "project_id": project_id,
        }

        memory_service._append_jsonl(audit_path, audit_record)

        # Verify audit record with metadata
        with open(audit_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        assert len(data) == 1
        record = data[0]
        assert record["operation"] == "memory_update"
        assert record["metadata"]["reason"] == "Content correction"
        assert record["metadata"]["approved_by"] == "admin_user"
        assert record["ip_address"] == "192.168.1.100"


class TestChangelog:
    """Test changelog functionality"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test data"""
        temp_path = tempfile.mkdtemp(prefix="test_changelog_")
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def memory_service(self, temp_dir):
        """Create MemoryService for testing"""
        return MemoryService(data_dir=temp_dir)

    def test_memory_changelog(self, memory_service):
        """Test memory changelog tracking"""
        tenant = "test_tenant"
        project_id = "test_project"
        memory_id = "mem_123"

        base = memory_service._store_dir(tenant, project_id)
        changelog_path = memory_service._dated_file(base, "changelog", "changes")

        # Add changelog entries
        changes = [
            {
                "memory_id": memory_id,
                "change_type": "created",
                "old_value": None,
                "new_value": "Initial memory content",
                "user_id": "user_456",
                "timestamp": datetime.utcnow().isoformat(),
                "project_id": project_id,
            },
            {
                "memory_id": memory_id,
                "change_type": "updated",
                "old_value": "Initial memory content",
                "new_value": "Updated memory content",
                "user_id": "user_456",
                "timestamp": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
                "project_id": project_id,
            },
        ]

        for change in changes:
            memory_service._append_jsonl(changelog_path, change)

        # Verify changelog
        assert os.path.exists(changelog_path)

        with open(changelog_path, "r", encoding="utf-8") as f:
            changelog_data = [json.loads(line) for line in f]

        assert len(changelog_data) == 2
        assert changelog_data[0]["change_type"] == "created"
        assert changelog_data[0]["new_value"] == "Initial memory content"
        assert changelog_data[1]["change_type"] == "updated"
        assert changelog_data[1]["old_value"] == "Initial memory content"
        assert changelog_data[1]["new_value"] == "Updated memory content"

    def test_changelog_with_versions(self, memory_service):
        """Test changelog with version tracking"""
        tenant = "test_tenant"
        project_id = "test_project"
        memory_id = "mem_456"

        base = memory_service._store_dir(tenant, project_id)
        changelog_path = memory_service._dated_file(base, "changelog", "changes")

        # Add versioned changes
        versions = [
            {
                "memory_id": memory_id,
                "version": 1,
                "change_type": "created",
                "content": "Version 1 content",
                "user_id": "user_456",
                "timestamp": datetime.utcnow().isoformat(),
                "project_id": project_id,
            },
            {
                "memory_id": memory_id,
                "version": 2,
                "change_type": "updated",
                "content": "Version 2 content",
                "user_id": "user_789",
                "timestamp": (datetime.utcnow() + timedelta(minutes=10)).isoformat(),
                "project_id": project_id,
            },
            {
                "memory_id": memory_id,
                "version": 3,
                "change_type": "updated",
                "content": "Version 3 content",
                "user_id": "user_456",
                "timestamp": (datetime.utcnow() + timedelta(minutes=20)).isoformat(),
                "project_id": project_id,
            },
        ]

        for version in versions:
            memory_service._append_jsonl(changelog_path, version)

        # Verify version tracking
        with open(changelog_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        assert len(data) == 3
        assert all(item["memory_id"] == memory_id for item in data)
        assert data[0]["version"] == 1
        assert data[1]["version"] == 2
        assert data[2]["version"] == 3
        assert data[2]["content"] == "Version 3 content"

import json
import os
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import patch

from app.services.memory_service import MemoryService

pytestmark = pytest.mark.asyncio

@pytest_asyncio.fixture
async def svc(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = {"data_root": str(data_dir), "embedding_model": "dummy"}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))

    with patch.dict(os.environ, {"PINAK_EMBEDDING_BACKEND": "dummy"}):
        service = MemoryService(config_path=str(config_path))
        await service.initialize()
        return service

async def test_quarantine_approve_semantic(svc):
    res = await svc.propose_memory(
        "semantic",
        {"content": "hello world", "tags": ["t1"]},
        tenant="default",
        project_id="pinak-memory",
        agent_id="agent-1",
        client_name="codex",
    )
    assert res["status"] == "pending"

    resolved = await svc.resolve_quarantine(
        res["id"],
        "approved",
        reviewer="admin",
        tenant="default",
        project_id="pinak-memory",
        agent_id="admin",
        client_name="admin",
    )
    assert resolved["status"] == "approved"


async def test_quarantine_reject(svc):
    res = await svc.propose_memory(
        "episodic",
        {"content": "event", "goal": "g", "outcome": "o"},
        tenant="default",
        project_id="pinak-memory",
        agent_id="agent-2",
        client_name="gemini",
    )
    resolved = await svc.resolve_quarantine(
        res["id"],
        "rejected",
        reviewer="admin",
        tenant="default",
        project_id="pinak-memory",
        agent_id="admin",
        client_name="admin",
    )
    assert resolved["status"] == "rejected"

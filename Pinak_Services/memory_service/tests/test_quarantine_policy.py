import os
import pytest
import pytest_asyncio
from unittest.mock import patch
from sqlalchemy import select, func

from app.services.memory_service import MemoryService
from app.core.schemas import ClientIssueCreate
from app.core.async_db import AsyncSessionLocal
from app.core.models import SemanticMemory

pytestmark = pytest.mark.asyncio

@pytest_asyncio.fixture
async def svc(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = {"data_root": str(data_dir)}

    with patch("app.services.memory_service.MemoryService._load_config", return_value=config):
        service = MemoryService()
        await service.initialize()
        return service

async def test_quarantine_auto_approve_trusted(svc, monkeypatch):
    monkeypatch.setenv("PINAK_QUARANTINE_AUTO_APPROVE", "1")
    monkeypatch.setenv("PINAK_TRUSTED_CLIENTS", "client-1")

    payload = {"content": "trusted semantic unique", "tags": ["t"]}
    res = await svc.propose_memory(
        "semantic",
        payload,
        "t1",
        "p1",
        client_id="client-1",
        client_name="trusted",
    )

    assert res.get("status") == "approved"

    # Verify directly in DB
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(func.count()).select_from(SemanticMemory).where(
                SemanticMemory.tenant == "t1",
                SemanticMemory.project_id == "p1",
                SemanticMemory.content == "trusted semantic unique"
            )
        )
        assert result.scalar() == 1


async def test_auto_resolve_issue_for_trusted(svc, monkeypatch):
    monkeypatch.setenv("PINAK_TRUSTED_CLIENTS", "client-2")
    monkeypatch.setenv("PINAK_AUTO_RESOLVE_ISSUES", "missing_client_id")

    issue = await svc.add_client_issue(
        ClientIssueCreate(error_code="missing_client_id", message="missing", layer=None),
        "t1",
        "p1",
        client_id="client-2",
        client_name="trusted",
    )

    assert issue.get("status") == "resolved"

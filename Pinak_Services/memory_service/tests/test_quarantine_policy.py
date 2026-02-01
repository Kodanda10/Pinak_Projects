import os
from unittest.mock import patch

from app.services.memory_service import MemoryService
from app.core.schemas import ClientIssueCreate


def test_quarantine_auto_approve_trusted(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = {"data_root": str(data_dir)}

    monkeypatch.setenv("PINAK_QUARANTINE_AUTO_APPROVE", "1")
    monkeypatch.setenv("PINAK_TRUSTED_CLIENTS", "client-1")

    with patch("app.services.memory_service.MemoryService._load_config", return_value=config):
        svc = MemoryService()
        payload = {"content": "trusted semantic", "tags": ["t"]}
        res = svc.propose_memory(
            "semantic",
            payload,
            "t1",
            "p1",
            client_id="client-1",
            client_name="trusted",
        )

        assert res.get("status") == "approved"

        with svc.db.get_cursor() as conn:
            cur = conn.execute(
                "SELECT count(*) FROM memories_semantic WHERE tenant = ? AND project_id = ?",
                ("t1", "p1"),
            )
            assert cur.fetchone()[0] == 1


def test_auto_resolve_issue_for_trusted(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = {"data_root": str(data_dir)}

    monkeypatch.setenv("PINAK_TRUSTED_CLIENTS", "client-2")
    monkeypatch.setenv("PINAK_AUTO_RESOLVE_ISSUES", "missing_client_id")

    with patch("app.services.memory_service.MemoryService._load_config", return_value=config):
        svc = MemoryService()
        issue = svc.add_client_issue(
            ClientIssueCreate(error_code="missing_client_id", message="missing", layer=None),
            "t1",
            "p1",
            client_id="client-2",
            client_name="trusted",
        )

        assert issue.get("status") == "resolved"

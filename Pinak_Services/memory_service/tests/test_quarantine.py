import json
import os
from pathlib import Path

from app.services.memory_service import MemoryService


def _make_config(tmp_path: Path) -> Path:
    config = {"data_root": str(tmp_path / "data"), "embedding_model": "dummy"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path


def test_quarantine_approve_semantic(tmp_path):
    config_path = _make_config(tmp_path)
    os.environ["PINAK_EMBEDDING_BACKEND"] = "dummy"
    service = MemoryService(config_path=str(config_path))

    res = service.propose_memory(
        "semantic",
        {"content": "hello world", "tags": ["t1"]},
        tenant="default",
        project_id="pinak-memory",
        agent_id="agent-1",
        client_name="codex",
    )
    assert res["status"] == "pending"

    resolved = service.resolve_quarantine(
        res["id"],
        "approved",
        reviewer="admin",
        tenant="default",
        project_id="pinak-memory",
        agent_id="admin",
        client_name="admin",
    )
    assert resolved["status"] == "approved"


def test_quarantine_reject(tmp_path):
    config_path = _make_config(tmp_path)
    os.environ["PINAK_EMBEDDING_BACKEND"] = "dummy"
    service = MemoryService(config_path=str(config_path))

    res = service.propose_memory(
        "episodic",
        {"content": "event", "goal": "g", "outcome": "o"},
        tenant="default",
        project_id="pinak-memory",
        agent_id="agent-2",
        client_name="gemini",
    )
    resolved = service.resolve_quarantine(
        res["id"],
        "rejected",
        reviewer="admin",
        tenant="default",
        project_id="pinak-memory",
        agent_id="admin",
        client_name="admin",
    )
    assert resolved["status"] == "rejected"

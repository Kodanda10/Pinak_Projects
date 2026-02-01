import os
from unittest.mock import patch

from app.services.memory_service import MemoryService
from app.core.schemas import MemoryCreate


def test_qmd_backend_disables_vectors(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = {"data_root": str(data_dir)}

    monkeypatch.setenv("PINAK_EMBEDDING_BACKEND", "qmd")
    with patch("app.services.memory_service.MemoryService._load_config", return_value=config):
        svc = MemoryService()
        res = svc.add_memory(MemoryCreate(content="hello qmd", tags=[]), "t1", "p1")

        assert res.content == "hello qmd"
        assert svc.vector_enabled is False

        # Keyword search should still return results
        hits = svc.search_hybrid("hello", "t1", "p1", limit=5)
        assert any("hello" in (h.get("content") or "") for h in hits)

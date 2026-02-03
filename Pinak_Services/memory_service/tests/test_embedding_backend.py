import os
import pytest
import pytest_asyncio
from unittest.mock import patch

from app.services.memory_service import MemoryService

pytestmark = pytest.mark.asyncio

def test_backend_none_skips_model_load(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = {"data_root": str(data_dir)}

    monkeypatch.setenv("PINAK_EMBEDDING_BACKEND", "none")
    with patch("app.services.memory_service.MemoryService._load_config", return_value=config):
        with patch("app.services.memory_service.SentenceTransformer") as mock_st:
            svc = MemoryService()
            assert svc.vector_enabled is False
            mock_st.assert_not_called()


async def test_update_delete_skip_vectors_when_disabled(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = {"data_root": str(data_dir)}

    monkeypatch.setenv("PINAK_EMBEDDING_BACKEND", "none")
    with patch("app.services.memory_service.MemoryService._load_config", return_value=config):
        svc = MemoryService()
        await svc.initialize() # Ensure DB ready
        svc.model.encode = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("encode called"))

        item = await svc.db.add_semantic("old", [], "t1", "p1", 111)
        assert await svc.update_memory("semantic", item["id"], {"content": "new"}, "t1", "p1") is True
        assert await svc.delete_memory("semantic", item["id"], "t1", "p1") is True

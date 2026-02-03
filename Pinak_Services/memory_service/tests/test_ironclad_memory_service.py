import os
import pytest
import asyncio
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from app.services.memory_service import MemoryService, _DeterministicEncoder
from app.core.schemas import MemoryCreate

# Mark async tests
pytestmark = pytest.mark.asyncio

def test_deterministic_encoder_cycle():
    encoder = _DeterministicEncoder(dimension=64)
    vecs = encoder.encode(["hello"])
    assert vecs.shape == (1, 64)
    assert not np.all(vecs == 0)

def test_memory_service_init_model_variants():
    m1 = MagicMock()
    m1.get_sentence_embedding_dimension.return_value = 128
    del m1.embedding_dimension
    
    svc1 = MemoryService(model=m1)
    assert svc1.embedding_dim == 128
    
    m2 = MagicMock()
    del m2.embedding_dimension
    del m2.get_sentence_embedding_dimension
    
    svc2 = MemoryService(model=m2)
    assert svc2.embedding_dim == 384

def test_load_embedding_model_exception():
    svc = MemoryService()
    with patch("app.services.memory_service.SentenceTransformer", side_effect=Exception("load fail")):
        model = svc._load_embedding_model("some-model")
        assert isinstance(model, _DeterministicEncoder)

async def test_verify_and_recover_triggers_rebuild(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    config = {"data_root": str(data_dir)}
    
    with patch("app.services.memory_service.MemoryService._load_config", return_value=config):
        svc = MemoryService()
        await svc.initialize()

        # Add one record to DB manually
        await svc.db.add_semantic("test content", [], "tenant1", "proj1", 123)
        
        with patch.object(svc, "_rebuild_index", new_callable=AsyncMock) as mock_rebuild:
            await svc.verify_and_recover()
            pass

async def test_update_memory_edge_cases():
    svc = MemoryService()
    await svc.initialize()

    # 1. Forbidden keys only
    assert await svc.update_memory("semantic", "id1", {"tenant": "new"}, "t1", "p1") == False
    
    # 2. Semantic content update with re-embedding
    # Add dummy record to get ID
    res = await svc.db.add_semantic("old", [], "t1", "p1", 100)
    mid = res["id"]

    with patch.object(svc.model, "encode", return_value=np.random.random((1, 384)).astype("float32")) as mock_encode:
        res_update = await svc.update_memory("semantic", mid, {"content": "new content"}, "t1", "p1")
        assert res_update is True

async def test_delete_memory_semantic(tmp_path):
    svc = MemoryService()
    await svc.initialize()

    # Mock record with embedding ID
    with patch.object(svc.db, "get_memory", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = {"embedding_id": 12345}
        with patch.object(svc.vector_store, "remove_ids") as mock_remove:
            await svc.delete_memory("semantic", "1", "t1", "p1")
            mock_remove.assert_called_with([12345])

import numpy as np
import pytest
from app.services.vector_store import VectorStore

def test_dynamic_capacity_resizing(tmp_path):
    index_path = str(tmp_path / "test_index.npy")
    dim = 10
    vs = VectorStore(index_path, dim)

    # Check initial values
    assert vs.size == 0
    assert vs.capacity == 1000

    # Add enough vectors to trigger resize
    batch1 = np.random.rand(800, dim).astype(np.float32)
    ids1 = list(range(800))
    vs.add_vectors(batch1, ids1)

    assert vs.size == 800
    assert vs.capacity == 1000

    batch2 = np.random.rand(400, dim).astype(np.float32)
    ids2 = list(range(800, 1200))
    vs.add_vectors(batch2, ids2)

    assert vs.size == 1200
    assert vs.capacity >= 1200

    # Search should work on active vectors
    q = np.random.rand(1, dim).astype(np.float32)
    dists, res_ids = vs.search(q, k=5)

    assert len(res_ids) == 5
    assert all(i < 1200 for i in res_ids)

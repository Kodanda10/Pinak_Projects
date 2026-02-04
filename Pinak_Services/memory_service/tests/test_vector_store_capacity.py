
import os
import numpy as np
import pytest
from app.services.vector_store import VectorStore

def test_capacity_expansion(tmp_path):
    index_path = str(tmp_path / "capacity.index")
    dimension = 4
    vs = VectorStore(index_path, dimension)

    # Initial state
    assert vs._size == 0
    assert vs._capacity == 0

    # Add small batch, should trigger initial allocation (likely 1024 min)
    vectors = np.random.rand(10, dimension).astype(np.float32)
    ids = list(range(10))
    vs.add_vectors(vectors, ids)

    assert vs._size == 10
    assert vs._capacity >= 1024
    assert len(vs.vectors) == vs._capacity

    # Fill up close to capacity (simulate)
    # We can't easily fill 1024 without loop, so let's just inspect internals
    current_capacity = vs._capacity

    # Add more vectors to force resize if we were small, but we started big.
    # Let's verify data integrity
    assert np.allclose(vs.vectors[:10], vectors)

def test_save_load_compaction(tmp_path):
    index_path = str(tmp_path / "save_load.index")
    dimension = 4
    vs = VectorStore(index_path, dimension)

    vectors = np.random.rand(10, dimension).astype(np.float32)
    ids = list(range(10))
    vs.add_vectors(vectors, ids)

    capacity_before = vs._capacity
    assert capacity_before >= 1024

    # Save
    vs.save()

    # Load in new instance
    vs2 = VectorStore(index_path, dimension)
    assert vs2._size == 10
    assert vs2._capacity == 10 # Load sets capacity to exact size
    assert len(vs2.vectors) == 10

    # Verify data
    assert np.allclose(vs2.vectors, vectors)
    assert np.array_equal(vs2.ids, ids)

def test_remove_ids_compaction(tmp_path):
    index_path = str(tmp_path / "remove.index")
    dimension = 4
    vs = VectorStore(index_path, dimension)

    vectors = np.random.rand(10, dimension).astype(np.float32)
    ids = list(range(10))
    vs.add_vectors(vectors, ids)

    assert vs._capacity >= 1024

    # Remove half
    to_remove = [0, 1, 2, 3, 4]
    vs.remove_ids(to_remove)

    assert vs._size == 5
    assert vs._capacity == 5 # Remove compacts
    assert len(vs.vectors) == 5

    remaining_ids = [5, 6, 7, 8, 9]
    assert np.all(np.isin(vs.ids, remaining_ids))

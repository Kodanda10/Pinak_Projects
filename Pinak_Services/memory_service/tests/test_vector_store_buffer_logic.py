
import pytest
import numpy as np
import os
import sys
# Ensure app import works
sys.path.append(os.path.join(os.getcwd(), 'Pinak_Services/memory_service'))
from app.services.vector_store import VectorStore

def test_search_with_buffered_data(tmp_path):
    """
    Verify that search works correctly when data is in the write buffer
    but not yet merged into the main index.
    """
    index_path = str(tmp_path / "buffered.index")
    dim = 4
    vs = VectorStore(index_path, dimension=dim)

    # 1. Add some vectors (should go to buffer)
    vecs1 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ], dtype=np.float32)
    ids1 = [1, 2]
    vs.add_vectors(vecs1, ids1)

    assert vs.ntotal == 2
    # Verify internal state (buffer should be used)
    assert len(vs._buffer_ids) == 1
    assert len(vs.ids) == 0

    # 2. Search immediately (should find buffered items)
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    dists, ids = vs.search(query, k=1)

    assert len(ids) == 1
    assert ids[0] == 1
    # dists[0] should be close to 0
    assert abs(dists[0]) < 1e-5

    # 3. Add more vectors
    vecs2 = np.array([
        [0.0, 0.0, 1.0, 0.0]
    ], dtype=np.float32)
    ids2 = [3]
    vs.add_vectors(vecs2, ids2)

    assert vs.ntotal == 3

    # 4. Search again (should search both buffers)
    query2 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    dists, ids = vs.search(query2, k=1)
    assert ids[0] == 3

    # 5. Force save (should merge)
    vs.save()
    assert len(vs._buffer_ids) == 0
    assert len(vs.ids) == 3

    # 6. Search after merge
    dists, ids = vs.search(query, k=1)
    assert ids[0] == 1

def test_hybrid_search_main_and_buffer(tmp_path):
    """
    Verify that search combines results from both the main index and the buffer.
    """
    index_path = str(tmp_path / "hybrid.index")
    dim = 2
    vs = VectorStore(index_path, dimension=dim)

    # Add to main (simulate by save)
    # [10, 10]
    vs.add_vectors(np.array([[10.0, 10.0]], dtype=np.float32), [10])
    vs.save()

    # Add to buffer
    # [1, 1]
    vs.add_vectors(np.array([[1.0, 1.0]], dtype=np.float32), [1])

    # Search for [1, 1]
    # Dist to [1,1] is 0.
    # Dist to [10,10] is high.
    query = np.array([1.0, 1.0], dtype=np.float32)
    dists, ids = vs.search(query, k=2)

    assert len(ids) == 2
    assert ids[0] == 1
    assert ids[1] == 10

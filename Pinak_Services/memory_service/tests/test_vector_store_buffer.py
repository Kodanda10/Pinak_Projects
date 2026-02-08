
import os
import pytest
import numpy as np
from app.services.vector_store import VectorStore

def test_vector_store_buffer_search(tmp_path):
    """Verify that items in the write buffer are found by search."""
    index_path = str(tmp_path / "buffer.index")
    vs = VectorStore(index_path, dimension=4)

    # Add vectors (should go to buffer)
    vec1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    id1 = [100]
    vs.add_vectors(vec1, id1)

    # Check that buffer is not empty (internal check, brittle but confirms logic)
    assert len(vs._buffer_vectors) == 1
    assert len(vs.ids) == 0  # Main storage empty

    # Search should find it (trigger flush)
    dists, ids = vs.search(vec1, k=1)
    assert ids == [100]
    assert np.isclose(dists[0], 0.0)

    # After search, buffer should be empty and main storage populated
    assert len(vs._buffer_vectors) == 0
    assert len(vs.ids) == 1

def test_vector_store_buffer_multiple_adds(tmp_path):
    """Verify multiple additions accumulate in buffer before flush."""
    index_path = str(tmp_path / "buffer_multi.index")
    vs = VectorStore(index_path, dimension=4)

    vec1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    vec2 = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    vs.add_vectors(vec1, [1])
    vs.add_vectors(vec2, [2])

    assert len(vs._buffer_vectors) == 2

    # Save should trigger flush
    vs.save()

    assert len(vs._buffer_vectors) == 0
    assert len(vs.ids) == 2
    assert 1 in vs.ids
    assert 2 in vs.ids

def test_vector_store_buffer_remove(tmp_path):
    """Verify remove flushes buffer before removing."""
    index_path = str(tmp_path / "buffer_remove.index")
    vs = VectorStore(index_path, dimension=4)

    vec1 = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    vs.add_vectors(vec1, [1])

    assert len(vs._buffer_vectors) == 1

    vs.remove_ids([1])

    assert len(vs._buffer_vectors) == 0
    assert len(vs.ids) == 0

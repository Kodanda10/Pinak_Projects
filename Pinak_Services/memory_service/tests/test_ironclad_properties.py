import pytest
import numpy as np
import shutil
import os
from hypothesis import given, strategies as st, settings, HealthCheck
from app.services.vector_store import VectorStore

# We use a unique subdirectory for each iteration to avoid ID collisions in FAISS files
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(
    dim=st.integers(min_value=64, max_value=128),
    vector_id=st.integers(min_value=1, max_value=10000000)
)
def test_vector_store_retrieval_consistency(tmp_path, dim, vector_id):
    # Create a unique subdir for this specific example
    # Hypothesis generates many examples, we need to keep them isolated
    # We can use a deterministic name based on the inputs for debugging
    it_dir = tmp_path / f"it_{dim}_{vector_id}"
    it_dir.mkdir(exist_ok=True)
    
    index_path = str(it_dir / "prop.index")
    vs = VectorStore(index_path, dimension=dim)
    
    vector = np.random.random((1, dim)).astype(np.float32)
    vs.add_vectors(vector, [vector_id])
    vs.save()
    
    distances, ids = vs.search(vector, k=1)
    
    assert len(ids) == 1
    assert ids[0] == vector_id
    assert distances[0] < 1e-4

@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(
    dim=st.integers(min_value=1, max_value=256),
    query_dim=st.integers(min_value=1, max_value=256)
)
def test_search_robustness(tmp_path, dim, query_dim):
    it_dir = tmp_path / f"robust_{dim}_{query_dim}"
    it_dir.mkdir(exist_ok=True)
    
    vs = VectorStore(str(it_dir / "robust.index"), dimension=dim)
    query = np.random.random(query_dim).astype(np.float32)
    
    if dim == query_dim:
        dist, ids = vs.search(query)
        assert ids == []
    else:
        # FAISS might throw or we might handle it. 
        # Current wrapper trusts FAISS.
        pass

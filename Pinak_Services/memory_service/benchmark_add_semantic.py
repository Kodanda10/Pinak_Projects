import time
import os
import numpy as np
from app.services.memory_service import MemoryService
from unittest.mock import MagicMock

def benchmark():
    # Mocking dependencies
    db_mock = MagicMock()
    db_mock.add_semantic.return_value = {"id": "123"}
    model_mock = MagicMock()
    model_mock.encode.return_value = np.array([[0.1] * 384], dtype=np.float32)

    # We will use the actual VectorStore to measure the impact
    from app.services.vector_store import VectorStore
    index_path = "test_memory_bench.npz"
    if os.path.exists(index_path):
        os.remove(index_path)
    if os.path.exists("test_memory_bench.npy"):
        os.remove("test_memory_bench.npy")

    vs = VectorStore(index_path, 384)

    num_memories = 1000
    print(f"Adding {num_memories} semantic memories WITH explicit save...")

    start = time.time()
    for i in range(num_memories):
        content = f"Test content {i}"
        embedding = model_mock.encode([content])[0]
        embedding_id = i
        vs.add_vectors(embedding, [embedding_id])
        vs.save()  # Explicit synchronous save

    end = time.time()
    print(f"Time WITH synchronous save: {end - start:.4f} seconds")

    if os.path.exists(index_path):
        os.remove(index_path)

    # Now benchmark WITHOUT synchronous save
    vs = VectorStore(index_path, 384)
    print(f"Adding {num_memories} semantic memories WITHOUT explicit save...")
    start = time.time()
    for i in range(num_memories):
        content = f"Test content {i}"
        embedding = model_mock.encode([content])[0]
        embedding_id = i
        vs.add_vectors(embedding, [embedding_id])
        # vs.save() # Rely on VectorStore's internal _schedule_save

    end = time.time()
    print(f"Time WITHOUT synchronous save: {end - start:.4f} seconds")

    if os.path.exists(index_path):
        os.remove(index_path)

if __name__ == "__main__":
    benchmark()

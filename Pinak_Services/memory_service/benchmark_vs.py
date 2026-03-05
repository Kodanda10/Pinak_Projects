import numpy as np
import time
from app.services.vector_store import VectorStore
import os

def benchmark():
    dimension = 384
    index_path = "test_benchmark.npz"
    if os.path.exists(index_path):
        os.remove(index_path)
    if os.path.exists("test_benchmark.npy"):
        os.remove("test_benchmark.npy")

    store = VectorStore(index_path, dimension)

    num_vectors = 10000
    batch_size = 1000
    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)
    ids = list(range(num_vectors))

    start = time.time()
    for i in range(0, num_vectors, batch_size):
        store.add_vectors(vectors[i:i+batch_size], ids[i:i+batch_size])
    end = time.time()
    print(f"Adding {num_vectors} vectors in batches of {batch_size} took {end - start:.4f} seconds")

    query = np.random.randn(1, dimension).astype(np.float32)
    start = time.time()
    for _ in range(100):
        store.search(query, k=10)
    end = time.time()
    print(f"100 searches took {end - start:.4f} seconds")

    if os.path.exists(index_path):
        os.remove(index_path)

if __name__ == "__main__":
    benchmark()

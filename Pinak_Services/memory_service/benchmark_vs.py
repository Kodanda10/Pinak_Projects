import time
import numpy as np
from app.services.vector_store import VectorStore
import os
import shutil

def main():
    dim = 384
    num_vectors = 100000
    test_dir = "test_benchmark_data"
    os.makedirs(test_dir, exist_ok=True)
    index_path = f"{test_dir}/vectors.npy"

    vs = VectorStore(index_path, dim)

    print(f"Adding {num_vectors} vectors...")

    vectors = np.random.rand(num_vectors, dim).astype(np.float32)
    ids = list(range(num_vectors))

    start_time = time.time()

    # Simulate adding vectors in small batches, like a real app might do
    batch_size = 100
    for i in range(0, num_vectors, batch_size):
        batch_vecs = vectors[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        vs.add_vectors(batch_vecs, batch_ids)
        if i % 10000 == 0:
            print(f"  Added {i} vectors...")

    end_time = time.time()
    print(f"Total time to add {num_vectors} vectors: {end_time - start_time:.2f} seconds")

    shutil.rmtree(test_dir)

if __name__ == "__main__":
    main()

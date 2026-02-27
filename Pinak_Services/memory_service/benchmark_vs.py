
import time
import numpy as np
import os
import sys

# Add the current directory to sys.path to allow importing app
sys.path.append(os.path.join(os.path.dirname(__file__)))

from app.services.vector_store import VectorStore

def benchmark_vector_store():
    # Configuration
    DIMENSION = 384
    NUM_VECTORS = 10000
    BATCH_SIZE = 1000
    QUERY_COUNT = 100
    INDEX_PATH = "benchmark_data/test_index"

    print(f"Benchmarking VectorStore with {NUM_VECTORS} vectors of dimension {DIMENSION}...")

    # Ensure clean state
    if os.path.exists(INDEX_PATH + ".npy"):
        os.remove(INDEX_PATH + ".npy")
    if os.path.exists(INDEX_PATH):
        os.remove(INDEX_PATH)

    os.makedirs("benchmark_data", exist_ok=True)

    try:
        # Initialize
        start_init = time.time()
        vs = VectorStore(INDEX_PATH, DIMENSION)
        print(f"Initialization: {time.time() - start_init:.4f}s")

        # Generate Data
        print("Generating random vectors...")
        vectors = np.random.rand(NUM_VECTORS, DIMENSION).astype(np.float32)
        ids = list(range(NUM_VECTORS))

        # Benchmark Add
        print(f"Adding {NUM_VECTORS} vectors in batches of {BATCH_SIZE}...")
        start_add = time.time()
        for i in range(0, NUM_VECTORS, BATCH_SIZE):
            batch_vecs = vectors[i : i + BATCH_SIZE]
            batch_ids = ids[i : i + BATCH_SIZE]
            vs.add_vectors(batch_vecs, batch_ids)
        end_add = time.time()
        print(f"Total Add Time: {end_add - start_add:.4f}s")
        print(f"Average Add Time per Vector: {(end_add - start_add) / NUM_VECTORS * 1000:.4f}ms")

        # Force Save
        start_save = time.time()
        vs.save()
        print(f"Save Time: {time.time() - start_save:.4f}s")

        # Benchmark Search
        print(f"Searching {QUERY_COUNT} random queries...")
        queries = np.random.rand(QUERY_COUNT, DIMENSION).astype(np.float32)

        start_search = time.time()
        for i in range(QUERY_COUNT):
            vs.search(queries[i])
        end_search = time.time()

        print(f"Total Search Time: {end_search - start_search:.4f}s")
        print(f"Average Search Time per Query: {(end_search - start_search) / QUERY_COUNT * 1000:.4f}ms")
        print(f"Queries per Second (QPS): {QUERY_COUNT / (end_search - start_search):.2f}")

    finally:
        # Cleanup
        if os.path.exists(INDEX_PATH + ".npy"):
            os.remove(INDEX_PATH + ".npy")
        if os.path.exists(INDEX_PATH):
            os.remove(INDEX_PATH)
        if os.path.exists("benchmark_data"):
            os.rmdir("benchmark_data")

if __name__ == "__main__":
    benchmark_vector_store()

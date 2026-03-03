import numpy as np
import time
import os
import sys

# Add parent dir to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'app')))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app.services.vector_store import VectorStore

def run_benchmark():
    if os.path.exists("test_bench.index"):
        os.remove("test_bench.index")
    vs = VectorStore("test_bench.index", 384)
    start = time.time()
    # Batch add is not used to simulate individual updates
    for i in range(5000):
        vs.add_vectors(np.random.random((1, 384)).astype(np.float32), [i])
    print(f"Time for 5000 inserts: {time.time() - start:.3f}s")
    if os.path.exists("test_bench.index"):
        os.remove("test_bench.index")

if __name__ == "__main__":
    run_benchmark()
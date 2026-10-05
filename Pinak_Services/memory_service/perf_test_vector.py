import time
import numpy as np
from app.services.vector_store import VectorStore
import os

vs = VectorStore("test_perf_vector.npy", 1536)
vs.add_vectors(np.random.rand(10000, 1536).astype(np.float32), list(range(10000)))

start = time.time()
q = np.random.rand(1, 1536).astype(np.float32)
for _ in range(100):
    vs.search(q)
end = time.time()
print("Time vector search:", end - start)

if os.path.exists("test_perf_vector.npy"):
    os.remove("test_perf_vector.npy")

import time
import numpy as np
from app.services.vector_store import VectorStore

vs = VectorStore("test_index", 384)
vs.reset()

start = time.time()
for i in range(1000):
    vs.add_vectors(np.random.rand(1, 384), [i])
print(f"Without batch: {time.time() - start:.4f}s")

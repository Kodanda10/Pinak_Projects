import sys
sys.path.append("Pinak_Services/memory_service")
from app.services.vector_store import VectorStore
import numpy as np
import time
import os

vs = VectorStore("test_index", 384)
vs.reset()

# Without batch
start = time.time()
for i in range(1000):
    vs.add_vectors(np.random.rand(1, 384), [i])
end = time.time()
print(f"Without real batch (vstack per add): {end - start:.4f}s")

# Wait, batch_add is currently yielding and doing nothing in VectorStore, so it uses vstack per add even inside batch_add block.
vs.reset()
start = time.time()
with vs.batch_add():
    for i in range(1000):
        vs.add_vectors(np.random.rand(1, 384), [i])
end = time.time()
print(f"Current batch_add (vstack per add): {end - start:.4f}s")

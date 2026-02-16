import numpy as np
import os
import sys

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Force the correct path used by MemoryService
index_path = "data/vectors.index.npy"

print(f"Loading index from {index_path}...")
try:
    data = np.load(index_path, allow_pickle=True).item()
    ids = data['ids']
    vectors = data['vectors']
    print(f"Total Vectors: {len(ids)}")
    print(f"IDs: {ids}")
    if len(vectors) > 0:
        print(f"Vector dim: {vectors.shape[1]}")
        print(f"Norms computed: {len(vectors)}")
except Exception as e:
    print(f"Error loading index: {e}")

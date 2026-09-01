## 2024-09-01 - Vector Store Buffer Bottleneck
**Learning:** The O(N) `np.vstack` in `add_vectors` within `VectorStore` causes significant performance degradation as the index grows because it constantly reallocates the array on every addition.
**Action:** Implemented an O(1) buffer list to temporarily hold newly added vectors, flushing them into the main `np.vstack` array only when a read operation (like `search`, `save`, `reconstruct`) necessitates up-to-date arrays.

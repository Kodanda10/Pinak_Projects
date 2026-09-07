## 2024-09-08 - O(1) buffering for VectorStore
**Learning:** The `VectorStore` implementation uses `np.vstack` for adding vectors one by one in `add_vectors`, which is an O(N) operation resulting in extreme performance bottlenecks for bulk additions (e.g., startup syncs or batch loads).
**Action:** Implement an O(1) in-memory buffer strategy using Python lists (e.g. `self._buffer_vectors.append(...)`), and only flush these buffers via `np.vstack` into the main array when a search, save, or rebuild operation explicitly requires it.

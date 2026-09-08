## 2025-02-23 - Buffer Optimization for Vector Store
**Learning:** The O(N) `np.vstack` in `VectorStore.add_vectors` creates a massive performance bottleneck because it copies the entire array upon every insert.
**Action:** Always implement an amortized O(1) list-based buffering strategy that accumulates individual additions and only flushes to the main O(N) array when required by operations like `search`, `save`, or `remove_ids`.

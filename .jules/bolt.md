## 2024-05-18 - Buffer numpy additions to prevent O(N) reallocation
**Learning:** The `VectorStore` class using Numpy arrays triggers an O(N) full array copy on every `np.vstack` during individual vector insertions, creating a severe bottleneck when processing long streams of individual adds.
**Action:** Implement O(1) list-based buffers to accumulate additions, flushing them to Numpy arrays only when search or save operations are invoked, effectively amortizing the reallocation cost.

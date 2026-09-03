## 2023-10-27 - O(N) Array Concatenation Bottleneck in VectorStore
**Learning:** Using `np.vstack` and `np.concatenate` on large numpy arrays for every single insertion creates an O(N) bottleneck that severely degrades performance as the index grows.
**Action:** Implement an O(1) list-based buffering strategy for new additions and only flush to the main O(N) numpy arrays when destructive operations (like search, save, or rebuild) require a complete state.

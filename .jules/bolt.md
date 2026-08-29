## 2025-02-28 - O(1) buffer optimization in VectorStore
**Learning:** Adding vectors iteratively directly to NumPy arrays involves O(N) memory allocations via `np.vstack`. In high-throughput operations (like memory insertion), this becomes a major bottleneck.
**Action:** When working with dynamically growing collections before processing/saving (like adding vectors before a commit or save), use standard Python lists (O(1) append) as intermediate buffers and flush them periodically or lazily (e.g., right before a search or save operation) using `np.vstack` in one go.

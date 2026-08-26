## 2025-02-28 - Bottleneck in numpy vector stacking
**Learning:** Using `np.vstack` for every single vector addition in `VectorStore` causes an O(N) performance bottleneck because it copies the entire array every time.
**Action:** Implement an O(1) list-based buffering strategy to accumulate additions, and only flush the buffer using `np.vstack` when a search, save, or other read operation is triggered.

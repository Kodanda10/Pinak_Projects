## 2024-05-23 - VectorStore np.vstack O(N^2) Scaling in batch_add
**Learning:** Using `np.vstack` to add individual vectors to a numpy array inside a loop (as done in `VectorStore.add_vectors`) creates an O(N^2) bottleneck because the entire array must be reallocated and copied for each addition.
**Action:** Always use memory buffering (e.g., Python lists) to collect vectors when executing batch operations inside a context manager like `VectorStore.batch_add`, then flush them in a single `np.vstack` and `np.concatenate` operation at the end to achieve O(N) performance.

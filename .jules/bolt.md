## 2024-06-21 - VectorStore Batch Insertion Bottleneck
**Learning:** The Numpy-based VectorStore replaced FAISS but maintained O(N^2) array copies during batch insertions because `add_vectors` iteratively called `np.vstack` even inside the `batch_add` context manager.
**Action:** Use thread-local buffering inside context managers to accumulate vectors and perform a single vectorized `np.vstack` at exit, changing O(N^2) to O(N).

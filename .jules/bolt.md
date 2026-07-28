## 2024-05-18 - [VectorStore Lazy Merge]
**Learning:** The VectorStore class in `Pinak_Services/memory_service/app/services/vector_store.py` buffers unmerged vectors in native Python lists (`_unmerged_vectors`, etc.) and lazily merges them via `np.vstack` prior to read/write operations, achieving amortized O(1) batch insertions and avoiding O(N²) complexity.
**Action:** Always maintain the lazy merging pattern when dealing with frequent appends in numpy-backed classes.

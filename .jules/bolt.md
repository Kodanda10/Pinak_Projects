## 2023-10-24 - Buffer vectors in VectorStore
**Learning:** Adding vectors iteratively by calling `np.vstack` causes O(N) array reallocation on every single insertion, creating a huge bottleneck when adding many vectors one by one.
**Action:** Implement a buffering list (`_vector_buffer`) to accumulate insertions and flush them in batches before reads (`search`, `reconstruct`, etc.) or saves.

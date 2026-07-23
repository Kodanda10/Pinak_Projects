## 2025-02-14 - VectorStore O(N^2) Insertion Bottleneck
**Learning:** Using `np.vstack` for every single vector insertion in `VectorStore.add_vectors` creates an O(N^2) bottleneck due to repeated array reallocation and copying. The memory service architecture relies heavily on this for real-time memory insertions.
**Action:** Buffer vectors in native Python lists (which have amortized O(1) appends) and lazily merge them with `np.vstack` only when a read/write operation (`search`, `save`, etc.) is requested.

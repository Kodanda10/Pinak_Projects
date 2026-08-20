## 2024-08-20 - Amortized O(1) Vector Insertion
**Learning:** The VectorStore implementation was doing O(N) array concatenation (np.vstack and np.concatenate) on every single vector addition, resulting in O(N^2) complexity for bulk insertions or high-throughput scenarios.
**Action:** Implemented a list-based buffering strategy (`_vectors_buffer`, `_ids_buffer`, `_norms_buffer`) that defers array concatenation until read operations (search, save, reconstruct), achieving amortized O(1) performance for insertions.

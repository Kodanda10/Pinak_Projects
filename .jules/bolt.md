## 2024-05-18 - VectorStore Reconstruct Bottleneck
**Learning:** Using `np.where(self.ids == vector_id)[0]` for searching by ID inside the VectorStore creates an O(N) linear scan that becomes a severe bottleneck for frequent reconstruct operations on large index arrays. The architecture heavily relies on these operations.
**Action:** Introduced an `id_to_idx` hash map to track indices on insertions and deletions, replacing the O(N) linear scan with an O(1) dictionary lookup, significantly improving performance.

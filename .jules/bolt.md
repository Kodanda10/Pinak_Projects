## 2024-08-13 - O(1) Amortized Vector Additions
**Learning:** The VectorStore implementation was doing O(N) reallocation via `np.vstack` for every single vector addition. When memory sizes get large, this creates an O(N^2) bottleneck for bulk ingestion.
**Action:** Use native python lists as a buffer (`_vectors_buffer.append(vectors)`) and flush with a single `np.vstack` during read/search operations. This achieves O(1) amortized insertions.

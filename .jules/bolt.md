## 2025-01-20 - VectorStore O(N^2) Insertion Bottleneck
**Learning:** `np.vstack` on every vector addition causes an O(N^2) bottleneck, leading to degraded performance during high-throughput insertions.
**Action:** Use a native Python list buffer (`_unmerged_vectors`) for amortized O(1) batch insertions and lazily merge via `np.vstack` only prior to read/write operations.

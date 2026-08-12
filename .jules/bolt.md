## 2024-05-24 - VectorStore O(N^2) Insertion Bottleneck
**Learning:** Using `np.vstack` on every insertion in `VectorStore` causes a full array copy, leading to O(N^2) complexity. This architecture makes bulk loading or frequent single insertions unnecessarily slow (amortized performance is ruined).
**Action:** Implemented a list-based buffering strategy (`_vectors_buffer`) to append inserts in O(1) time and flush them via a single `np.vstack` only during reads or saves, reducing 5000 insertions from ~3.5s to ~0.03s.

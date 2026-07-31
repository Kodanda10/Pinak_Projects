## 2025-08-29 - VectorStore O(N) Insertion Bottleneck
**Learning:** `np.vstack` is O(N). Doing it on every insert for a growing VectorStore creates severe O(N^2) scaling issues when ingesting many individual memories. For 20,000 insertions, it goes from 0.3s to 14.0s.
**Action:** Always buffer individual array appends in native Python lists (amortized O(1)) and lazily merge via `np.vstack` only before reads (search, save, etc.).

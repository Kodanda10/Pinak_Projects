## 2026-02-28 - VectorStore O(N^2) Array Allocation Fix
**Learning:** Appending arrays to continuous `np.empty` sequences using `np.vstack` and `np.concatenate` triggers an O(N) array reallocation on every element addition resulting in a crippling O(N^2) total allocation algorithm constraint.
**Action:** Always allocate array properties initialized natively with a size 'capacity' value and track active segments using a 'size' parameter. Resizing elements dynamically by a power of 2 when reaching capacity yields an amortized O(1) insertion scaling pattern.

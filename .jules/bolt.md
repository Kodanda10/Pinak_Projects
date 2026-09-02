## 2025-02-28 - O(1) List Buffering for Vector Insertions
**Learning:** Sequential vector additions using `np.vstack` create an O(N) bottleneck because they re-allocate and copy the entire array on every single insertion.
**Action:** Always use an O(1) list-based buffering strategy (e.g., Python lists with `.append()`) and only flush/consolidate into the main numpy array when necessary (e.g., before search or save operations).

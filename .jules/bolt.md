## 2024-05-24 - O(N^2) Complexity in NumPy Vector Appends
**Learning:** Calling `np.vstack` and `np.concatenate` directly inside `add_vectors` for every new vector addition creates an O(N^2) bottleneck because the arrays must be fully copied on each insertion. This severely degrades performance when appending many vectors sequentially.
**Action:** Implement a list-based buffering strategy (`_vectors_buffer`, `_ids_buffer`, `_norms_buffer`) to append elements in O(1) time and perform a single batched flush via `np.vstack` during reads or saves.

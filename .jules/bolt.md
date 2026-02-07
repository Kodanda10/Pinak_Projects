## 2024-05-22 - NumPy vstack Bottleneck
**Learning:** `np.vstack` creates a full copy of the array. Doing this in a loop (e.g., adding vectors one by one) results in $O(N^2)$ complexity.
**Action:** Always buffer small NumPy arrays in a list and use `np.vstack` or `np.concatenate` once on the batch (amortized $O(1)$).

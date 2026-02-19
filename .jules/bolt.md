## 2025-02-19 - Vector Store Resizing Optimization
**Learning:** `np.vstack` in a loop creates a new array and copies all data on every iteration, leading to $O(N^2)$ complexity. Pre-allocating and dynamically resizing (amortized $O(N)$) is significantly faster.
**Action:** Use dynamic array resizing pattern for accumulators instead of repeated concatenation.

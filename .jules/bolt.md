## 2026-02-25 - VectorStore Performance
**Learning:** Naive use of `np.vstack` for adding vectors creates an O(N²) bottleneck due to repeated copying.
**Action:** Use amortized O(1) dynamic array resizing (pre-allocation + capacity doubling) for append-heavy workloads in NumPy.

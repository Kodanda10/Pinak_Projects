## 2024-05-18 - [VectorStore Performance Bottleneck]
**Learning:** Using `np.vstack` for sequential or streaming insertions into a numpy array creates an $O(N^2)$ bottleneck because a new array is allocated and copied on every insert.
**Action:** Use native Python lists to buffer incoming vectors and apply a single `np.vstack` flush operation during reads or saves to achieve amortized $O(1)$ performance.

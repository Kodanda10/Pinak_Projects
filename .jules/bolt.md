## 2024-05-22 - Vector Search Optimization
**Learning:** Python's `numpy.argsort` on large arrays is significantly slower than `numpy.argpartition` for top-k selection. Also, calculating Euclidean distance using `(a-b)^2` involves large temporary array allocations which can be avoided using the expanded form `a^2 + b^2 - 2ab` and matrix multiplication, provided vector norms are precomputed.
**Action:** Use `argpartition` for top-k. Use dot product + precomputed norms for Euclidean distance in high-dimensional vector search.

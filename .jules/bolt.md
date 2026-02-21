## 2024-05-23 - VectorStore Optimization
**Learning:** `np.vstack` and `np.concatenate` in a loop are performance killers ($O(N^2)$). Replacing them with pre-allocated arrays and dynamic resizing (amortized $O(1)$) yielded a ~23x speedup for 200k vectors.
**Action:** Always check for repeated array concatenations in data ingestion loops. Use dynamic array resizing pattern.

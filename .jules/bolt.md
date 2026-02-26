## 2025-05-24 - [VectorStore Dynamic Resizing]
**Learning:** `np.vstack` and `np.concatenate` inside a loop are deceptively expensive ($O(N^2)$) for building arrays. Pre-allocating a buffer and resizing geometrically (amortized $O(1)$) yielded a ~2x speedup for 2000 vectors and prevents memory fragmentation.
**Action:** Always use pre-allocation or list accumulation before `np.array()` when size is unknown, or implement dynamic resizing classes for streaming data. Avoid `np.append`/`np.stack` in hot loops.

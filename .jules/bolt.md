# Bolt's Journal

## 2025-05-18 - [VectorStore Allocation Bottleneck]
**Learning:** Initializing numpy arrays on every `add_vectors` call (using `vstack`/`concatenate`) leads to O(N²) complexity for sequential additions, which severely impacts performance (from 2000+ to <1500 vectors/sec as N grows).
**Action:** Use amortized resizing (doubling capacity) and pre-allocated arrays. Treat `self.vectors` as a buffer with a `capacity` and an active `_size`. Slice operations `[:self._size]` ensure correctness while maintaining O(1) amortized insertion.

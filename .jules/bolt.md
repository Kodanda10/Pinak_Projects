## 2024-05-24 - Buffered NumPy insertions
**Learning:** Using `np.vstack` for every single vector addition in NumPy results in O(N²) memory usage and massive slowdowns because NumPy arrays are contiguous blocks of memory and must be fully reallocated and copied on every single insert.
**Action:** Always buffer small inserts in a native Python list and periodically convert and flush them with a single `np.vstack` to achieve amortized O(1) performance.

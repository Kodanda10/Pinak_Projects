## 2024-04-10 - [VectorStore L2 Distance Optimization]
**Learning:** In `VectorStore.search()`, calculating L2 distances in NumPy using `.flatten()` and intermediate array allocations causes significant overhead.
**Action:** A fully vectorized approach using `.ravel()` for a 1D view, 1D `np.dot` for norms, and in-place array modifications (`-=` and `out=`) drastically improves performance. To prevent `UFuncTypeError` during in-place operations, ensure the 1D query array is explicitly cast to the target array's dtype (e.g., `.astype(self.vectors.dtype, copy=False)`).

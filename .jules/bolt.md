## 2025-02-24 - VectorStore.search() Performance Optimization
**Learning:** In `VectorStore.search()`, calculating L2 distances in NumPy using `.flatten()` and intermediate array allocations causes significant overhead.
**Action:** A fully vectorized approach using `.ravel()` for a 1D view, 1D `np.dot` for norms, and in-place array modifications (`-=` and `out=`) drastically improves performance.

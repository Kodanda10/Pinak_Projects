## 2025-03-05 - NumPy Distance Calculation In-Place Optimization
**Learning:** In `VectorStore.search()`, calculating L2 distances in NumPy using `.flatten()` and intermediate array allocations causes significant overhead.
**Action:** Use a fully vectorized approach using `.ravel()` for a 1D view, 1D `np.dot` for norms, and in-place array modifications (`-=` and `out=`) to drastically improve performance.

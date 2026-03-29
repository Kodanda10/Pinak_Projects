## 2024-05-24 - NumPy Vector Store Search Optimization
**Learning:** In `VectorStore.search()`, calculating L2 distances in NumPy using `.flatten()` and intermediate array allocations causes significant overhead.
**Action:** Use a fully vectorized approach with `.ravel()` for a 1D view, 1D `np.dot` for norms, and in-place array modifications (`-=` and `out=`) to drastically improve performance and reduce memory allocations.

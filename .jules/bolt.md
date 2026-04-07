## 2025-01-20 - [Performance Learnt]
**Learning:** Python's built-in `hash()` function is used for `embedding_id` generation in `MemoryService`, which is non-deterministic across Python runs due to hash randomization and poses a consistency/reloading risk for the vector index.
**Action:** Always use a deterministic hash function like `hashlib.md5()` or `hashlib.sha256()` when generating IDs for persistent storage.

## 2025-01-20 - [L2 Norm Performance in VectorStore]
**Learning:** In `VectorStore.search()`, calculating L2 distances in NumPy using `.flatten()` and intermediate array allocations causes significant overhead. A fully vectorized approach using `.ravel()` for a 1D view, 1D `np.dot` for norms, and in-place array modifications (`-=` and `out=`) drastically improves performance.
**Action:** When implementing mathematical operations on large arrays in NumPy, avoid `.flatten()` (which creates a copy) and prefer `.ravel()` or direct vectorized operations.

## 2025-01-20 - [Batch Vector Additions in VectorStore]
**Learning:** When incrementally building large NumPy arrays for batch vector additions, using `np.vstack` repeatedly is O(N^2) and extremely slow. Caching additions in a Python list and performing a single `np.concatenate(vlist, axis=0)` is orders of magnitude faster.
**Action:** Always batch array concatenations when appending sequentially.

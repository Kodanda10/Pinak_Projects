## 2024-08-17 - VectorStore List Buffering
**Learning:** Using `np.vstack` directly on a numpy array for single record insertions results in an O(N^2) time complexity because numpy copies the entire array each time.
**Action:** Instead, append single additions to native Python list buffers (`_vectors_buffer`, `_ids_buffer`, `_norms_buffer`). Defer flushing these buffers to the numpy arrays (`np.vstack`, `np.concatenate`) until a read or save operation occurs, achieving amortized O(1) inserts.

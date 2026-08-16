## 2023-10-27 - VectorStore amortized O(1) inserts

**Learning:** The Numpy-based `VectorStore` was using `np.vstack` for every single insertion. Because `np.vstack` copies the entire array into a new one, inserting N items one by one turns into an O(N^2) operation, making insertion extremely slow when the index gets large.

**Action:** Implemented an amortized O(1) strategy where incoming vectors, IDs, and norms are appended to standard Python lists (`_vectors_buffer`, `_ids_buffer`, `_norms_buffer`). These lists are flushed into the main numpy arrays in a single `np.vstack`/`np.concatenate` operation only when necessary (e.g., during `search`, `save`, or `remove_ids`). This significantly speeds up insertions without breaking thread safety or compatibility.

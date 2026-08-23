## 2025-02-23 - O(1) List-based Buffering for Numpy Array Additions
**Learning:** Using `np.vstack` repeatedly in `VectorStore.add_vectors` creates an O(N) performance bottleneck because numpy arrays are reallocated and copied on every insert.
**Action:** Implement O(1) additions by buffering new arrays in native Python lists (e.g. `_vector_buffer.append(vectors)`), and only `vstack`/`concatenate` them when searching or saving (`_flush_buffers()`).

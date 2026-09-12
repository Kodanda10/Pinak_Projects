## 2024-05-24 - O(N) np.vstack Bottleneck
**Learning:** Using `np.vstack` directly on every `add_vectors` call in `VectorStore` creates a significant O(N) performance bottleneck because NumPy reallocates memory for the entire array on each concatenation.
**Action:** Implemented an amortized O(1) list-based buffering strategy by appending vectors to a Python list (`_vector_buffer`) and flushing them periodically before operations that require the full array (like search or save). Next time, proactively use list buffers for frequent vector accumulations instead of immediate `np.vstack`.

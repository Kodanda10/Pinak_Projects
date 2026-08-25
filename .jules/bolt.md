## 2024-08-25 - Replace np.vstack with List Buffering in VectorStore
**Learning:** Using `np.vstack` in the inner loop of `add_vectors` creates an O(N) operation per batch because it must copy the entire growing array. In `Pinak_Services/memory_service/app/services/vector_store.py`, this became a severe bottleneck when appending many small batches of vectors.
**Action:** Implemented an O(1) list-based buffering strategy (`_vector_buffer`, `_id_buffer`) and delayed array concatenation until operations requiring full array structure (like `search` or `save`) are invoked via `_flush_buffers()`.

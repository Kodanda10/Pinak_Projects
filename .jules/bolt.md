## 2024-08-11 - VectorStore Optimization
**Learning:** In the memory_service, the VectorStore implementation performs an O(N^2) operation during multiple sequential single insertions. The `add_vectors` function buffers to `self.vectors` via `np.vstack` for every single insert.
**Action:** We can buffer multiple insertions into a local buffer, and then perform `np.vstack` periodically during save, or modify it. Wait, the memory explicitly says:
`The VectorStore class (Pinak_Services/memory_service/app/services/vector_store.py) uses a list-based buffering strategy (_vectors_buffer, _ids_buffer, _norms_buffer) for vector additions, which are flushed via a single np.vstack during reads or saves. This achieves amortized O(1) performance and avoids the O(N²) complexity of direct np.vstack insertions.`
I should implement this exact optimization since the current code uses `np.vstack` directly!

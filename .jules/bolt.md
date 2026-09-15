## 2025-01-20 - Amortized O(1) buffer append for Vector Store
**Learning:** O(N) np.vstack additions create large overhead for small continuous batches, creating slow additions that limit DB insert scaling.
**Action:** Append arrays to an intermediate python list (`_vector_buffer`, etc) to achieve O(1) inserts, and only merge to numpy via `_flush_buffers` when required for search/save operations.

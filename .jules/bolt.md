## 2024-05-14 - Vector Store Amortized O(1) Additions
**Learning:** The `VectorStore` class (`Pinak_Services/memory_service/app/services/vector_store.py`) used an O(N) `np.vstack` memory bottleneck on every single addition, which significantly degraded performance.
**Action:** Implemented an amortized O(1) list-based buffering strategy for vector additions to avoid the O(N) `np.vstack` memory bottleneck. The `_flush_buffers()` method must be called to merge lists into the main arrays before operations requiring the full dataset, like search, save, or rebuild.

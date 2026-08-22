## 2024-05-30 - O(N^2) VectorStore insertions mitigated by O(1) buffers
**Learning:** The `VectorStore` class (`Pinak_Services/memory_service/app/services/vector_store.py`) directly used `np.vstack` for every single insertion. This leads to an O(N^2) time complexity during batch/streaming additions since every addition rewrote the entire multi-dimensional array.
**Action:** Always verify if continuous single-item inserts on large NumPy arrays can be converted to an O(1) list-based appending strategy with lazy batch concatenation (a flush buffer pattern) when real-time updates are needed.

## 2024-10-25 - Explicit Save vs Debounced Save in Memory Service
**Learning:** The `MemoryService` was explicitly calling `vector_store.save()` after every write operation, negating the `VectorStore`'s internal debounced save mechanism (`_schedule_save`). This resulted in $O(N)$ disk I/O for every insertion.
**Action:** Always check if the underlying storage class implements its own persistence strategy before adding explicit persistence calls in the service layer.

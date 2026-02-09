## 2024-05-23 - Thread Storm in Vector Store
**Learning:** The `VectorStore` was creating a new `threading.Timer` thread for every `add_vectors` call to debounce saves. This resulted in thousands of threads being created and destroyed during bulk ingestion (O(N) thread creation).
**Action:** Always check `timer.is_alive()` before creating a new timer thread for background tasks. Prefer throttling (one active timer) over debouncing (cancelling and recreating) for high-frequency events where intermediate execution is acceptable or eventual consistency is handled.

## 2026-05-19 - Thread-Local Batch Context Optimization
**Learning:** The `VectorStore.batch_add` context manager suffered from O(N^2) memory reallocation penalties because batch additions were not buffered.
**Action:** When implementing batch context managers (e.g., `VectorStore.batch_add`), use thread-local storage (`threading.local()`) to buffer data during the `yield` block, then flush with single `np.vstack` operations to achieve O(N) performance and avoid race conditions.

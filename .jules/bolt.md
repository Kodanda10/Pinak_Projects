## 2024-10-25 - Amortized O(1) List Buffering over NumPy vstack
**Learning:** Using `np.vstack` and `np.concatenate` directly during vector addition in `VectorStore` incurs an O(N) penalty per insertion, which acts as a major performance bottleneck for frequent memory updates.
**Action:** Implement list-based buffering for vector additions to achieve amortized O(1) performance. Remember to dynamically sum buffer sub-array lengths (e.g., `sum(len(b) for b in buffer)`) for count operations and defer flushing into the main arrays until global state is strictly needed (like `search` or `save`).

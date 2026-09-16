## 2025-01-20 - Buffered Vector Store Additions
**Learning:** Using `np.vstack` directly on every vector addition is an O(N) operation and causes severe memory bottlenecking as the store grows, especially under batch workloads.
**Action:** Implement amortized O(1) list-based buffering (`_vector_buffer.append()`) for additions, and only flush (`_flush_buffers()`) to the main numpy arrays when operations requiring the full dataset (like `search` or `save`) are invoked.

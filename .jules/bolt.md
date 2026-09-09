## 2024-03-24 - O(N) np.vstack Bottleneck in VectorStore

**Learning:** The Pinak Memory Service `VectorStore` uses an `np.vstack` for every single vector addition. While `np.vstack` copies the entire array every time, creating an O(N) performance cliff as the vector store grows. This memory service architecture is particularly sensitive to this because vector additions happen frequently in loops/batches.

**Action:** Implemented an amortized O(1) list-based buffering strategy for vector additions. Vectors, IDs, and norms are appended to standard Python lists first, and only combined into the main numpy arrays using `np.vstack`/`np.concatenate` during a `_flush_buffers` operation before operations that require the full dataset like `search`, `save`, `remove_ids`, `reconstruct`, and `ntotal`. This significantly improves insertion throughput.

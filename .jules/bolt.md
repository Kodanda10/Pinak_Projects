## 2024-09-10 - VectorStore O(N) append bottleneck

**Learning:** Continuous O(N) array reallocation (`np.vstack`, `np.concatenate`) in memory-intensive operations (like adding vectors to the `VectorStore`) acts as a massive bottleneck as data grows.
**Action:** Implement amortized O(1) list-based buffering strategies for operations requiring frequent appends, flushing buffers into main arrays only when required for reads or saves.

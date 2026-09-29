## 2024-09-29 - VectorStore O(N) Insertion Bottleneck
**Learning:** Using `np.vstack` and `np.concatenate` for every single vector insertion in the VectorStore causes an O(N) memory reallocation bottleneck, which significantly degrades performance as the index grows.
**Action:** Implement a list-based buffering strategy (`_vector_buffer`, `_id_buffer`, `_norm_buffer`) to accumulate insertions and flush them in bulk before reads or saves, reducing expensive NumPy array copying.

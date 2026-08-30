# Bolt Journal
## 2024-08-30 - O(1) buffer optimization for Numpy Vector Store
**Learning:** Using `np.vstack` for continuous addition of vectors in `VectorStore.add_vectors` creates an O(N) bottleneck that scales poorly with index size, causing significant slowdowns during large sequential or frequent batch additions.
**Action:** Always prefer O(1) list appends (e.g., `list.append()`) to buffer new records before periodically flushing them with `np.vstack` when performing operations that require a unified array like search or save.

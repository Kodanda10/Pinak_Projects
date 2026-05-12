## 2026-05-12 - Optimize VectorStore batch additions
**Learning:** Naive np.vstack calls in a loop cause O(N^2) performance penalties. Thread-local storage buffers can completely avoid this overhead by grouping np.vstack and np.concatenate, handling nested calls correctly.
**Action:** Use threading.local() to create buffers for batched additions when implementing context managers to handle concurrency safely without locking bottlenecks.

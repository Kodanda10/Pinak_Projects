## 2024-05-24 - [Batch Context Manager with Thread-Local Accumulation]
**Learning:** `VectorStore.batch_add` was just a simple yield that saved on exit, but inside the batch block, `add_vectors` still called `np.vstack` for every single addition, taking O(N^2) time for large batch loads.
**Action:** Use thread-local storage inside the context manager to buffer all incoming vectors to standard Python lists during the block, then execute a single O(N) `np.concatenate` and `np.vstack` upon context exit.

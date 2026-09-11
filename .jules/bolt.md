## 2024-05-24 - O(N) np.vstack Bottleneck in VectorStore Amortization
**Learning:** Calling `np.vstack` for every single insert into an empty numpy array creates a massive O(N) bottleneck, taking ~8s for 2000 insertions of 10-vector batches.
**Action:** Implementing amortized O(1) list-based buffering for vector additions prior to flushing avoids this bottleneck entirely, reducing 2000 insertions from ~8s to ~0.78s. `len(buffer)` incorrectly returns number of arrays not items, so `sum(len(b) for b in buffer)` must be used for sizing calculations like `ntotal`.

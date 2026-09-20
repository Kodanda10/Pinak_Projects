## 2024-05-24 - List-Based Buffering for Vector Additions
**Learning:** `np.vstack` for growing arrays by a small amount is an O(N) operation and a known performance bottleneck for vector additions in Python.
**Action:** Use list-based buffering to amortize vector appending to O(1) amortized, merging the buffer on operations that read the full array like search, save, and reconstruct. Note `len(buffer)` does not equal the number of added vectors if appending chunks; use `sum(len(b) for b in buffer)`.

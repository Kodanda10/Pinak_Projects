## 2024-12-14 - List-based buffering for NumPy arrays
**Learning:** O(N) np.vstack on every addition creates a massive performance bottleneck as the index grows.
**Action:** Implemented amortized O(1) list-based buffering to accumulate vectors, IDs, and norms, and only merge them into the main NumPy arrays before destructive or read operations (like save, search, reconstruct).

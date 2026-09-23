## 2024-10-24 - NumPy vstack O(N) Bottleneck
**Learning:** Calling `np.vstack` for every single vector addition in an iterative process creates an O(N^2) memory reallocation bottleneck, which can severely impact latency as the dataset grows.
**Action:** Implement list-based buffering to accumulate arrays, and flush them to the main NumPy array in a single batch operation only when necessary (e.g. before search, save, or remove).

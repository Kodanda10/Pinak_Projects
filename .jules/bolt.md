## 2025-02-12 - Vector Store Performance
**Learning:** `np.vstack` for appending vectors in a loop is an $O(N^2)$ anti-pattern. Using pre-allocated arrays with dynamic resizing (capacity doubling) reduces insertion time from quadratic to amortized constant time.
**Action:** Always pre-allocate numpy arrays or use dynamic resizing when building collections incrementally. Avoid `np.concatenate` or `np.vstack` inside loops.

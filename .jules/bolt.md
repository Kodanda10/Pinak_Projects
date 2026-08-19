## 2024-08-19 - NumPy O(N²) reallocation bottleneck

**Learning:** Reallocating and copying numpy arrays on every individual insertion using `np.vstack` scales at O(N²), causing severe performance bottlenecks during burst workloads (e.g. 10,000 insertions took ~7.3s instead of ~0.2s). Native Python lists are significantly faster for buffering due to amortized O(1) appends.

**Action:** When accumulating data dynamically, always buffer insertions into native Python lists and defer the expensive conversion to a continuous NumPy array (via `np.vstack` or `np.concatenate`) until absolutely necessary (e.g., during read operations or persisting to disk).

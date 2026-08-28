## 2025-02-28 - O(1) Buffering Strategy for NumPy Arrays
**Learning:** Using `np.vstack` directly during frequent, small array insertions (like individual vector additions) creates a severe O(N) performance bottleneck because NumPy reallocates memory and copies the entire array every time.
**Action:** Always use an O(1) list-based buffering strategy (e.g., appending to a standard Python list `self._buffer.append(item)`) and perform a single batch `np.vstack` flush operation before read-heavy paths (search, save). Never repeatedly mutate NumPy arrays dynamically in performance-critical paths.

## 2024-05-23 - Threading Timer vs Throttling
**Learning:** `threading.Timer` debouncing (cancel + start) is extremely expensive for high-frequency updates, as it creates a new thread for every event.
**Action:** Use throttling (check `timer.is_alive()` and return) instead of debouncing for background save operations. This avoids thread spam and can yield massive performance gains (e.g., 38x speedup).

## 2024-05-23 - Dynamic Array Resizing
**Learning:** When managing multiple parallel arrays (e.g., vectors, ids, norms), ensure ALL of them use dynamic resizing. Optimizing only the largest array (vectors) while leaving others to use `np.concatenate` ($O(N)$) will shift the bottleneck, negating performance benefits.
**Action:** Implement consistent capacity management for all related arrays in a data structure.

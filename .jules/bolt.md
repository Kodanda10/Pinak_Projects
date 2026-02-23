## 2025-02-18 - NumPy Dynamic Array Anti-Pattern
**Learning:** Using `np.vstack` or `np.concatenate` to append to a NumPy array in a loop (e.g., `self.vectors = np.vstack([self.vectors, new_vector])`) is an O(N²) operation because it copies the entire array on every addition. This creates a severe bottleneck even for small datasets (e.g., 5000 items).
**Action:** Use a pre-allocated array with a `size` counter and resize geometrically (e.g., double capacity) when full. This achieves amortized O(1) insertion.

## 2025-02-18 - Threading Timer Spam
**Learning:** In a debounced save mechanism using `threading.Timer`, simply canceling and restarting the timer on every event (`add_vectors`) can spawn thousands of threads per second during batch operations, overwhelming the scheduler.
**Action:** Use a "throttle" pattern instead: check `if self.timer.is_alive()` and return immediately if true. This ensures data is saved periodically without thread thrashing.

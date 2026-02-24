## 2024-05-23 - [Thread Creation Overhead in High-Frequency Loops]
**Learning:** Python's `threading.Timer` spawns a new thread. Creating threads in a tight loop (e.g., inside an `add_vectors` method) introduces massive overhead (context switching, OS allocation).
**Action:** Instead of cancelling and recreating timers for debouncing, use a "check if alive" pattern to throttle or batch operations. This reduced execution time for 20k vector additions from 6.6s to 0.4s (combined with dynamic array resizing).

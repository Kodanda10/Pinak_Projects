## 2024-08-14 - Thread Creation Overhead in Debouncers
**Learning:** In `VectorStore`, scheduling a save by cancelling and recreating a `threading.Timer` on every single vector addition caused massive overhead (~360ms for 1000 items). While amortized O(1) buffer insertions via `append` (instead of `np.vstack`) helped somewhat, the thread thrashing was the primary bottleneck.
**Action:** When debouncing high-frequency operations, always check if a timer is already alive before cancelling and recreating it, rather than blindly restarting the timer on every call.

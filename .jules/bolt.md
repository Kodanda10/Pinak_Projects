## 2024-04-09 - [Thread Spam during Batch Processing]
**Learning:** Python's `threading.Timer.cancel()` does not prevent the underlying OS thread from being created and starting; it only prevents the payload from executing. Using it for debouncing high-frequency events (like batch adding vectors) leads to massive thread creation overhead and slows down processing significantly (e.g., 3x slower for 10k items).
**Action:** Use throttling (checking if the timer is `None` before creating a new one) instead of debouncing for high-frequency operations that spawn OS threads.

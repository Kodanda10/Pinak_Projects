## 2025-05-19 - Database Connection Overhead
**Learning:** `DatabaseManager.get_cursor()` creates a new `sqlite3.connect` and closes it for every single database operation. In SQLite, opening and closing connections is expensive (0.09s vs 0.003s for 1000 operations).
**Action:** Implement thread-local connection reuse in `DatabaseManager.get_cursor()` to dramatically reduce query latency.

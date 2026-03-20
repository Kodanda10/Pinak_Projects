
## 2024-03-01 - [Thread-Local SQLite Connections]
**Learning:** Creating and tearing down SQLite connections for every query using `sqlite3.connect()` creates a huge performance bottleneck in isolated environments that perform heavy transactional database calls.
**Action:** Replace direct ad-hoc `connect()` instantiations with a `threading.local()` cache/pool which shares a connection instance per-thread. Make sure to close cursor inside transaction's `finally` block instead of closing the entire connection.

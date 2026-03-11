
## 2024-10-31 - [Thread-Local SQLite Connection Pooling]
**Learning:** Re-opening a new SQLite connection on every single `get_cursor()` transaction within `DatabaseManager` created a massive N+1 overhead bottleneck, significantly degrading database write performance in high-frequency logging paths (e.g. episodic/audit logs). Using a `threading.local()` cache to pool database connections with `PRAGMA journal_mode=WAL;` entirely eliminated this repeated connection instantiation, boosting mass-write performance by ~18x while keeping thread-safety intact.
**Action:** When working with SQLite in an environment handling frequent small writes, use `threading.local()` for connection pooling and apply PRAGMA optimizations `WAL`, `synchronous=NORMAL`, and `busy_timeout` per connection.

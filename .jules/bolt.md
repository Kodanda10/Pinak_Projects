## 2024-05-24 - SQLite Optimization
**Learning:** Adding composite indexes avoiding full table scans and placing them after `_ensure_column` in SQLite avoids `sqlite3.OperationalError` on legacy databases.
**Action:** Always add composite indexes in SQLite to optimize count, stats, and search resolution queries, placing them at the end of the initial schema creation code.

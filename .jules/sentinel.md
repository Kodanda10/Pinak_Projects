## 2025-02-14 - Fix SQL Injection in DatabaseManager.update_memory
**Vulnerability:** Possible SQL injection vector through string-based query construction when interpolating dictionary keys directly into the SET clause in `DatabaseManager.update_memory`.
**Learning:** Even if the input dictionary (`updates`) originates from an application layer that filters invalid keys, dynamic SQL query generation from dict keys at the DB layer is unsafe without local sanitation, leading to potential SQL injection.
**Prevention:** Validate dict keys using `str.isidentifier()` before using them to dynamically construct SQL query strings like `SET {clause}`, and log any critical codebase-specific security learnings.

## 2024-05-16 - Prevent Dynamic SQL Injection in `update_memory`
**Vulnerability:** Found a critical SQL injection vector in `DatabaseManager.update_memory()` where unsanitized keys from the `updates` dictionary were directly interpolated into an `UPDATE` SQL `SET` clause without validation.
**Learning:** Even when parameterized queries are used for the *values*, dynamically generating the SQL structure (like column names) from unvalidated input allows attackers to manipulate the query structure.
**Prevention:** Always validate dynamically provided column names using strict allowlists or structural validation like Python's `.isidentifier()`, and explicitly forbid updating system-managed columns like IDs or tenant boundaries.

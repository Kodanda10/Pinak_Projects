## 2025-02-14 - Prevent SQL Injection in Dynamic Query Construction
**Vulnerability:** SQL injection in `DatabaseManager.update_memory` via dynamically building the `UPDATE` set clause using raw dictionary keys (`updates.keys()`).
**Learning:** Using raw user-provided dictionaries to construct SQL queries, even with parameterized values, leaves the column names vulnerable to injection if not sanitized.
**Prevention:** Validate all dynamically injected column names using `if not isinstance(key, str) or not key.isidentifier(): raise ValueError(...)` before incorporating them into the SQL string.

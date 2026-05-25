## 2025-02-14 - Prevent SQL Injection via Dictionary Keys in Dynamic Queries
**Vulnerability:** A SQL injection and mass assignment vulnerability existed in `DatabaseManager.update_memory` due to unsafe construction of the `UPDATE` query's `SET` clause directly from unvalidated dictionary keys (`updates.items()`).
**Learning:** Dynamic query generation that concatenates keys from user-provided or external dictionaries into SQL statements is highly dangerous, even if the parameter values are safely parameterized using `?`.
**Prevention:** Always validate that dictionary keys used in dynamic SQL query construction are valid strings and Python identifiers (e.g., using `isinstance(key, str)` and `key.isidentifier()`) before building the query.

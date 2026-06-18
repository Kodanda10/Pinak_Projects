## 2026-06-18 - SQL Injection via Dictionary Keys in dynamic SET clause
**Vulnerability:** SQL Injection in `DatabaseManager.update_memory` where unsanitized dictionary keys were directly interpolated into the `SET` clause of an `UPDATE` statement.
**Learning:** Application-level key filters do not inherently prevent SQL injection if the underlying database layer dynamically constructs queries from dictionary keys without validation.
**Prevention:** Always validate dynamically interpolated column names or dictionary keys using `str.isidentifier()` at the database query execution layer.

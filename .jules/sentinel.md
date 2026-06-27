## 2026-06-27 - Prevent SQL Injection in Dynamic UPDATE Queries
**Vulnerability:** SQL Injection in DatabaseManager.update_memory due to dynamically interpolating unsanitized dictionary keys into the SET clause.
**Learning:** Application-level key filters (like forbidden_keys) do not prevent SQL injection or mass assignment if the underlying database layer dynamically constructs queries from unsanitized dictionary keys.
**Prevention:** Always validate dictionary keys using str.isidentifier() at the database query execution layer.

## 2024-06-16 - Prevent SQL Injection via Dictionary Keys
**Vulnerability:** SQL injection vulnerability in `DatabaseManager.update_memory` caused by dynamically interpolating unsanitized dictionary keys into the SET clause of an UPDATE statement.
**Learning:** Application-level key filters do not prevent SQL injection or mass assignment if the underlying database layer dynamically constructs queries from unsanitized dictionary keys.
**Prevention:** Always validate dynamically interpolated dictionary keys using `str.isidentifier()` at the database query execution layer to prevent SQL injection.

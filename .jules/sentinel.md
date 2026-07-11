## 2026-07-11 - Prevent SQL Injection via Dynamic Column Names
**Vulnerability:** SQL injection vulnerability in `DatabaseManager.update_memory` where dynamic dictionary keys were interpolated directly into the `SET` clause of the SQL UPDATE statement without validation.
**Learning:** Application-level key filters (like `forbidden_keys` in `memory_service.py`) do not prevent SQL injection or mass assignment if the underlying database layer (`app/core/database.py`) dynamically constructs queries from unsanitized dictionary keys.
**Prevention:** Always validate dictionary keys using `str.isidentifier()` at the database query execution layer before string formatting them into SQL queries.

## 2024-06-12 - Prevent SQL Injection via Mass Assignment
**Vulnerability:** SQL injection vulnerability found in `update_memory` where unsanitized dictionary keys from `updates` were directly concatenated into the `SET` clause of the SQL UPDATE query.
**Learning:** The database correctly maps table names to prevent injection there, but dynamically building the `SET` clause using arbitrary user-provided JSON/dictionary keys without validation exposes the database to SQL injection attacks and mass assignment.
**Prevention:** Enforce strict validation on all keys used in dynamic SQL queries using `isinstance(key, str)` and `key.isidentifier()` to ensure they are valid SQL column names before constructing the query string.

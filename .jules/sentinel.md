## 2024-05-18 - SQL Injection in `update_memory`
**Vulnerability:** The `update_memory` function dynamically interpolates unsanitized dictionary keys from the `updates` parameter into the SQL `SET` clause, allowing SQL injection and mass assignment.
**Learning:** Application-level key filters don't prevent SQL injection at the database layer if dynamic queries are built from unsanitized input.
**Prevention:** Always validate dictionary keys using `str.isidentifier()` or strict allowlists at the database query execution layer.

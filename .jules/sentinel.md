## 2024-05-24 - SQL Injection in Dynamic Update Queries
**Vulnerability:** SQL injection vulnerability in `update_memory` via unsanitized dictionary keys used to construct the `SET` clause.
**Learning:** Application-level key filtering is insufficient if the database layer dynamically constructs queries from unsanitized input keys.
**Prevention:** Always validate dynamically constructed SQL identifiers (like column names) using `str.isidentifier()` at the database query execution layer.

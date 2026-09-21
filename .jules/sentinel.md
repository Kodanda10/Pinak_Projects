## 2023-09-21 - SQL Injection in Dynamic UPDATE Query Construction
**Vulnerability:** The `update_memory` function in `database.py` was vulnerable to SQL injection because it dynamically constructed the `SET` clause of an `UPDATE` query using keys from a user-provided `updates` dictionary without any validation or sanitization.
**Learning:** Even when parameterized queries are used for values, dynamically inserting dictionary keys directly into a query string can lead to structural SQL injection (e.g. providing a key like `name = 'Hacked', id`).
**Prevention:** Always validate that dynamically constructed column names match expected patterns or use Python's `.isidentifier()` to ensure they are valid alphanumeric identifiers before inclusion in query strings.

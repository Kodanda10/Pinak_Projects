## 2024-05-24 - SQL Injection in Dynamic Column Names
**Vulnerability:** The `update_memory` method in `DatabaseManager` dynamically generated SQL `SET` clauses from dictionary keys without sanitizing them, allowing SQL injection attacks via malicious dictionary keys.
**Learning:** Always validate dynamically generated SQL column names against strict allowlists or regexes (e.g., `^[a-zA-Z0-9_]+$`), as parameter binding (`?`) only protects the *values*, not the *column names*.
**Prevention:** Enforce strict alphanumeric validation on all dynamically generated column names before interpolating them into SQL queries.

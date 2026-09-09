## 2024-05-18 - Prevent SQL Injection in Dynamic UPDATE Queries
**Vulnerability:** SQL injection vulnerability in `DatabaseManager.update_memory` because dictionary keys used for column names in the `SET` clause were directly injected without escaping.
**Learning:** Dynamic generation of SQL query parts like column names from untrusted dictionary keys can lead to SQL injection if not properly sanitized and quoted, even when using parameterized queries for the values.
**Prevention:** Always explicitly quote dynamically provided column names (e.g. using `"{}"` in SQLite) and escape internal double quotes when generating SQL queries from dictionary structures.

## 2024-10-05 - SQL Injection via Dynamic Query Parameters
**Vulnerability:** The `update_memory` function dynamically constructed a SQL `UPDATE` statement's `SET` clause using unvalidated dictionary keys provided by the caller. This allowed injection of arbitrary SQL statements via maliciously crafted keys.
**Learning:** Even when using parameterized queries for values, dynamically constructing query structures (like column names in a `SET` clause) from unvalidated user input introduces critical SQL injection vulnerabilities.
**Prevention:** Always validate dynamically generated column names against an allowlist or ensure they conform to strict identifier rules (e.g., `isidentifier()`). Never trust keys from user-provided dictionaries when building SQL queries.

## 2024-05-24 - SQL Injection in Dynamic UPDATE
**Vulnerability:** SQL Injection via unsanitized dictionary keys in dynamic UPDATE statements.
**Learning:** Python dictionary keys can be a vector for SQL injection if they are directly formatted into a query string, even when the corresponding values are parameterized.
**Prevention:** Always validate dictionary keys using `str.isidentifier()` (or against an allowlist) when dynamically constructing SQL clauses, even when values are parameterized.

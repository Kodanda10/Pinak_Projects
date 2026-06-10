## 2026-06-10 - Fix SQL Injection in update_memory
**Vulnerability:** SQL injection via unsanitized dictionary keys used to construct UPDATE queries.
**Learning:** Dynamic query construction using dictionary keys requires strict validation to ensure keys are valid identifiers.
**Prevention:** Always validate that dynamic column names are valid strings and Python identifiers using `key.isidentifier()` before using them in SQL queries.

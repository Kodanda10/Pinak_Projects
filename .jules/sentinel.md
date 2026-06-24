## 2026-06-24 - SQL Injection in Dynamic UPDATE clause
**Vulnerability:** SQL injection in `DatabaseManager.update_memory` due to dynamic interpolation of unsanitized dictionary keys into the SET clause.
**Learning:** The application validated values, but not the keys used to dynamically construct the SET clause, allowing arbitrary column updates (mass assignment) or SQL logic injection (e.g. `content = 'hacked', id`).
**Prevention:** Always validate dynamically provided dictionary keys against an allowlist or using strict identifier checks like `str.isidentifier()` before using them as SQL column names.

## 2026-07-25 - Prevent SQL Injection in Dynamic Column Names
**Vulnerability:** SQL injection vulnerability in `update_memory` due to unsanitized user-provided column names being directly interpolated into the `UPDATE` query's `SET` clause.
**Learning:** Direct interpolation of user keys into SQL statements without validation allows arbitrary SQL execution, even when parameterized for values.
**Prevention:** Always validate dynamically constructed column names using `str.isidentifier()` to ensure they are valid SQL identifiers before string interpolation.

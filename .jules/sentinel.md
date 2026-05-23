## 2026-05-23 - SQL Injection via Dynamic Query Construction
**Vulnerability:** `update_memory` used unsanitized dictionary keys from arbitrary user JSON input to construct the `SET` clause of an `UPDATE` statement.
**Learning:** Even if values are parameterized (`?`), unsanitized column names (keys) used in f-strings for dynamic query generation introduce SQL injection and mass assignment risks.
**Prevention:** Always validate dynamic column names against a strict allowlist or ensure they are valid identifiers using `isidentifier()` before incorporating them into a SQL statement.

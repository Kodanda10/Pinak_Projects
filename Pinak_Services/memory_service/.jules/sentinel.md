## 2025-01-20 - Discovery

## 2026-08-02 - SQL Injection in Update Method
**Vulnerability:** SQL injection found in `update_memory` where user-provided dictionary keys were directly interpolated into the `SET` clause of a `UPDATE` statement.
**Learning:** Even when using parameterized queries for values (`?`), constructing the SQL string using unvalidated column names (keys from the input dictionary) allows an attacker to manipulate the SQL statement.
**Prevention:** Always validate that dynamically provided column names strictly conform to expected identifier patterns (e.g., using `isidentifier()`) or check against an allowlist before interpolating them into SQL statements.

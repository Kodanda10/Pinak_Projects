## 2024-08-12 - SQL Injection in Dynamic SET Clauses
**Vulnerability:** A critical SQL injection vulnerability was found in `DatabaseManager.update_memory` where unvalidated dictionary keys were interpolated directly into the `SET` clause of an `UPDATE` statement via `f"{k} = ?"`.
**Learning:** Even when values are parameterized, dynamically generating query structure (like column names) from untrusted inputs without proper quoting or validation allows an attacker to inject SQL logic.
**Prevention:** Always strictly quote dynamic identifiers (like column or table names) with double quotes in SQLite (e.g., `\"{k}\"`), or validate them against a strict allowlist before interpolation.

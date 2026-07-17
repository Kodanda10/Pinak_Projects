## 2025-02-18 - Prevent SQL Injection in Dynamic Column Updates
**Vulnerability:** SQL injection vulnerability in `DatabaseManager.update_memory` caused by dynamically constructing the `SET` clause from unfiltered dictionary keys in `updates.items()`.
**Learning:** Even when using parameterized queries (e.g., `?` bindings for values), using untrusted dictionary keys directly to build column names (e.g., `f"{k} = ?"`) allows attackers to inject arbitrary SQL logic into the column specification.
**Prevention:** Always validate dynamically provided column names using `str.isidentifier()` to ensure they contain only valid characters and cannot include SQL syntax.

## 2025-10-13 - SQL Injection in Memory Updates
**Vulnerability:** SQL injection vulnerability in `DatabaseManager.update_memory` caused by directly injecting dictionary keys into the `SET` clause of a SQL `UPDATE` statement.
**Learning:** Keys of dynamic dictionary payloads must never be directly interpolated into SQL queries without sanitization, as they allow manipulation of the SQL query structure and updating of protected fields like `id` and `tenant` (IDOR).
**Prevention:** Always validate dynamically generated column names using strict regex or `.isidentifier()` checks, and explicitly exclude protected fields from being modified by user-provided payloads.

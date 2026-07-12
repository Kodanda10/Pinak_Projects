## 2025-02-19 - Fix SQL injection in DatabaseManager update method
**Vulnerability:** The SQLite query execution in `DatabaseManager.update_memory` interpolated unvalidated dictionary keys directly into the `UPDATE` statement `SET` clauses.
**Learning:** This repo builds up SQL statements dynamically using unsanitized string templating, leaving it vulnerable to injection attacks if the `updates` dict ever comes from untrusted API inputs.
**Prevention:** Always validate dynamically-inserted structural identifiers (table or column names) with strict criteria like `str.isidentifier()` to prevent SQL injection vulnerabilities that parameter bindings can't cover.

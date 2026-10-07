## 2026-10-07 - Dynamic Query Parameter SQL Injection
**Vulnerability:** SQL injection and potential tenant isolation bypass via dynamic dictionary keys in the `update_memory` UPDATE statement.
**Learning:** Formatting keys directly from user-controlled payload dictionaries into SQL statement clauses (like `SET k = ?`) is vulnerable if keys aren't strictly validated.
**Prevention:** Always validate dynamic keys explicitly (e.g. using `.isidentifier()`) and ensure protected fields are excluded from updates at the database layer.

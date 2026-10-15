## 2023-10-15 - SQL Injection in Dynamic Column Updates
**Vulnerability:** SQL Injection in `update_memory` due to unvalidated dictionary keys concatenated into `UPDATE` statements.
**Learning:** Dynamic column updates using user-provided dict keys (`set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])`) without `isidentifier()` validation or allow-listing allows SQL injection and unauthorized updates to protected fields (like `id`, `tenant`).
**Prevention:** Always validate dynamic dictionary keys using `.isidentifier()` and explicitly exclude protected fields from being updated to ensure tenant isolation and prevent SQL injection.

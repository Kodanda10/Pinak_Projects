## 2026-07-07 - Prevent SQL injection in update_memory
**Vulnerability:** SQL injection vulnerability in `DatabaseManager.update_memory` due to dynamic interpolation of unsanitized dictionary keys into the SET clause.
**Learning:** Application-level key filters (like `forbidden_keys`) do not prevent mass assignment or SQL injection if the underlying database layer constructs queries from unsanitized keys.
**Prevention:** Validate all dynamically constructed column names (e.g., using `str.isidentifier()`) before interpolating them into SQL strings at the database query execution layer.

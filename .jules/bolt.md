## 2023-10-25 - Safe Composite Multi-Tenant Index Creation
**Learning:** In the memory_service, tests simulate legacy schemas where certain columns (e.g. `tenant`, `project_id`) might not exist yet. Blindly adding `CREATE INDEX` for composite keys causes `sqlite3.OperationalError` during tests because the columns aren't there.
**Action:** When adding composite indexes to a schema initialization script that must support legacy tables, always check that every involved column actually exists using `self._column_exists()` before creating the index.

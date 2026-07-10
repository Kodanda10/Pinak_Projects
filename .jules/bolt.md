## 2026-07-10 - SQLite Composite Index Optimization
**Learning:** Single-column indexes (like `status`) in multi-tenant SQLite schemas lead to inefficient linear scans when querying across `client_id`, `tenant`, and `project_id`.
**Action:** Always create composite indexes (e.g., `(client_id, tenant, project_id, status)`) for multi-tenant queries and ensure new indexes are renamed when using `CREATE INDEX IF NOT EXISTS` so the new structure is applied.

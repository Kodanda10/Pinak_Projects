## 2025-08-30 - Added Composite Indexes to Logging Tables
**Learning:** The `logs_client_issues` and `memory_quarantine` SQLite tables were using less optimal single-column indexes on `status` for queries that filter by `(client_id, tenant, project_id, status)`, leading to inefficient lookups.
**Action:** Use composite indexes on all frequently filtered columns to enable COVERING INDEX usage and prevent suboptimal single-index usage or full table scans.

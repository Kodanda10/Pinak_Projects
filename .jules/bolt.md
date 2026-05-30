## 2025-05-30 - Add indexes for count_client_issues and count_quarantine
**Learning:** Database performance optimizations: The `logs_client_issues` and `memory_quarantine` SQLite tables utilize composite indexes on `(client_id, tenant, project_id, status)` to prevent full table scans and significantly speed up count queries.
**Action:** When adding indexes to optimize queries, create composite indexes on all filtered columns.

## 2026-06-16 - Add composite indexes for count queries
**Learning:** The `logs_client_issues` and `memory_quarantine` SQLite tables currently only have single-column indexes, lacking composite indexes on `(client_id, tenant, project_id, status)` which are frequently used in count queries.
**Action:** Add composite indexes to speed up multi-column filtering in count queries.

## 2024-06-05 - Optimize count queries in DatabaseManager
**Learning:** `count_client_issues` and `count_quarantine` in `DatabaseManager` filter by `client_id`, `tenant`, `project_id`, and `status`. Without composite indexes on these columns, these queries result in full table scans, taking ~1.5s for 100 queries on a 50k row table.
**Action:** Created composite indexes `idx_logs_client_issues_count` and `idx_memory_quarantine_count` on `(client_id, tenant, project_id, status)` for these tables. This reduced query time from 1.5s to 0.12s for 100 queries (a >90% reduction).

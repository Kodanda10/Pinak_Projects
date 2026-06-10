## 2024-03-24 - Database Indexes on Frequently Queried Fields
**Learning:** `logs_client_issues` and `memory_quarantine` tables are frequently queried with a composite filter `(client_id, tenant, project_id, status)`. Adding a composite index reduces query time significantly from ~0.024s to ~0.006s for 100k records.
**Action:** Always look for `COUNT(*)` queries with multiple `WHERE` clauses as candidates for composite indexes to avoid full table scans.

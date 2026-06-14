## 2024-06-15 - Add missing composite indexes for count queries
**Learning:** The count queries for `logs_client_issues` and `memory_quarantine` tables were doing full table scans due to missing composite indexes on `(client_id, tenant, project_id, status)`.
**Action:** Always verify if database indices mentioned in codebase knowledge or documentation actually exist in the database schema.

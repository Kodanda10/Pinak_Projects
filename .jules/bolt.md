## 2025-08-29 - Missing composite indexes on multi-tenant tables
**Learning:** In a multi-tenant application using SQLite, queries filtering by `tenant` and `project_id` along with a trailing column (like `status` or `ts`) experience significant performance degradation (e.g. 18s vs 1.6s) if only single-column indexes exist for the trailing column. Single-column indexes cannot optimize the isolation columns.
**Action:** Always append composite indexes (e.g., `(tenant, project_id, status)`) to heavily queried multi-tenant tables to prevent full table scans and N+1 queries.

## 2024-05-24 - Missing composite indexes on multi-tenant tables
**Learning:** Multi-tenant applications require composite indexes that include the tenant and project_id columns to avoid full table scans when querying large datasets.
**Action:** Always add composite indexes (e.g., `(tenant, project_id, status)` or `(tenant, project_id, ts)`) when querying multi-tenant tables to ensure optimal query performance.

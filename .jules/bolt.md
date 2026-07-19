## 2024-07-20 - Multi-Tenant Index Optimization
**Learning:** Using single-column indexes on highly selective fields (like `status`) in multi-tenant tables causes inefficient linear scans when querying by `client_id`, `tenant`, and `project_id`. When adding new composite indexes, you must rename the index and explicitly `DROP INDEX IF EXISTS` the old one to avoid leaving unused structures.
**Action:** Always create composite covering indexes containing `(client_id, tenant, project_id, [selective_field])` for multi-tenant queries.

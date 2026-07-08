## 2024-07-08 - Composite Indexes for Multi-Tenant Queries
**Learning:** Single-column indexes on highly selective multi-tenant tables (like `status`) cause inefficient linear scans for the remaining tenant filters (`client_id`, `tenant`, `project_id`).
**Action:** Replace single-column indexes with composite indexes on `(client_id, tenant, project_id, selective_field)` for multi-tenant SQLite queries to enable covering index usage and prevent linear scans.

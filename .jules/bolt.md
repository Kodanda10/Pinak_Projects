## 2024-05-24 - Composite indexes for multi-tenant SQLite tables
**Learning:** When optimizing multi-tenant queries in SQLite that filter by `client_id`, `tenant`, `project_id`, and a selective field (like `status`), use composite or covering indexes incorporating all these fields. A single-column index causes inefficient linear scans for the remaining filters.
**Action:** Always create composite indexes on `(client_id, tenant, project_id, [status/layer])` for tables where multi-tenant isolation and specific client querying are standard.

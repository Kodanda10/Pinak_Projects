## 2024-05-25 - Composite Indexes for Multi-Tenant Queries
**Learning:** When optimizing multi-tenant queries in SQLite that filter by `client_id`, `tenant`, `project_id`, and a selective field (like `status` or `embedding_id`), using a single-column index causes inefficient linear scans for the remaining filters.
**Action:** Always use composite covering indexes incorporating all these fields to prevent inefficient single-column index scans.

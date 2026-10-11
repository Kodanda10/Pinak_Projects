## 2025-02-23 - Composite Indexes on Multi-Tenant Logs
**Learning:** In heavily queried multi-tenant architectures relying on SQLite, single-column indexes on status or timestamps are insufficient when queries always filter by `tenant` and `project_id`. This causes full table scans or poor index selectivity, making multi-tenant queries O(N).
**Action:** Always add composite indexes starting with `(tenant, project_id)` followed by the commonly filtered/sorted columns on multi-tenant tables.

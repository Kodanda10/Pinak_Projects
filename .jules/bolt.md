## 2025-02-09 - Added Composite Indexes for Multi-Tenant Queries
**Learning:** In a multi-tenant memory service, frequently accessed logs and quarantine tables lacking composite indexes (`tenant`, `project_id`, `ts` / `status`) result in full table scans when filtered by tenant, severely degrading read performance at scale.
**Action:** Always add composite indexes starting with multi-tenant isolation columns (tenant, project_id) followed by frequently queried status or timestamp fields on heavily queried operational tables.

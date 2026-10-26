## 2024-10-26 - Adding composite indexes for multi-tenant isolation
**Learning:** In a multi-tenant system, missing composite indexes on tenant and project_id can lead to N+1 queries and full table scans.
**Action:** Always append composite database indexes covering multi-tenant isolation columns `(tenant, project_id)` and status or timestamp on heavily queried tables to optimize filtering.

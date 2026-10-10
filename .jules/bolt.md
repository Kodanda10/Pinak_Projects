## 2023-10-10 - Missing Database Indexes on Multi-Tenant Views
**Learning:** In a multi-tenant application where every row is isolated by `tenant` and `project_id`, failing to include composite indexes on these fields for heavily queried tables (logs_access, memory_quarantine, etc.) causes severe performance bottlenecks (N+1 queries/full table scans) as datasets grow.
**Action:** Always verify that multi-tenant filter columns (`tenant`, `project_id`) are included in composite database indexes for any table routinely queried via list/fetch operations.

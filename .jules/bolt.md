## 2024-08-01 - Missing Multi-Tenant Indexes Cause Table Scans
**Learning:** In the Pinak Memory Service architecture, querying memory tables by `embedding_id` also filters by `tenant` and `project_id`. Without composite indexes, SQLite performs full table scans (O(N)), which severely bottlenecks multi-tenant vector hydration for large tables.
**Action:** Always ensure large tables queried by tenant or project include composite indexes `(tenant, project_id, ...)` to guarantee O(log N) lookups instead of O(N) scans.

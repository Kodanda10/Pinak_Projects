## 2024-11-09 - Missing Composite Indexes on Multi-tenant Tables
**Learning:** Heavily-queried multi-tenant tables (`logs_access`, `memory_quarantine`, `clients_registry`, `logs_agents`) lacking composite indexes `(tenant, project_id, ...)` caused severe N+1 query performance degradation and full table scans.
**Action:** Always append composite indexes combining the multi-tenant isolation columns `(tenant, project_id)` with temporal or status fields to multi-tenant tables.

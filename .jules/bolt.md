## 2025-02-18 - Missing Composite Indexes
**Learning:** Heavily queried multi-tenant tables like logs_access, memory_quarantine, clients_registry, logs_agents, and logs_client_issues are lacking multi-tenant composite indexes, which could result in N+1 queries and full table scans.
**Action:** Append the new composite indexes for `tenant` and `project_id` combined with specific query criteria for each table, ensuring that I don't replace the existing single-column indexes, to prevent performance regressions for single-column searches.

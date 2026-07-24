## 2024-05-18 - Multi-tenant indexing in SQLite
**Learning:** In a multi-tenant SQLite application, single column indexes are insufficient. Most queries filter by `tenant` and `project_id`. Missing composite indexes causes full table scans.
**Action:** Always create composite indexes starting with `tenant` and `project_id` for multitenancy filtering (e.g., `(tenant, project_id, embedding_id)`) while retaining existing single column indexes if they are still required elsewhere for sorting or disjoint querying.

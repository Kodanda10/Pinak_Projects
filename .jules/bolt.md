## 2026-07-11 - SQLite Multi-Tenant Filtering Anti-Pattern
**Learning:** Single-column indexes on highly selective fields (like `status`) or lack of indexes for vector search result in O(N) full table scans during multi-tenant queries because SQLite cannot effectively combine filters across unindexed `tenant` and `project_id` columns.
**Action:** Always create composite indexes containing all filtering columns, such as `(embedding_id, tenant, project_id)` and `(client_id, tenant, project_id, status)`, to ensure efficient covering index usage.

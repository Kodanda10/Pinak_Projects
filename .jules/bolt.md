## 2024-05-24 - Missing Multi-Tenant Composite Indexes
**Learning:** Found a performance bottleneck where multi-tenant queries were performing full table scans because tables like `memories_semantic`, `memories_episodic`, and `memories_rag` lacked composite indexes on `(tenant, project_id)`. SQLite needs these composite indexes to efficiently filter rows before applying further conditions.
**Action:** Always create composite indexes starting with tenant and project_id for tables in a multi-tenant architecture to prevent O(N) query performance degradation as the database grows.

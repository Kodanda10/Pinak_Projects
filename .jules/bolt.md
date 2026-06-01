## 2025-02-18 - Missing Indexes on Memory Tables
**Learning:** The memory tables lacked a composite index on `(embedding_id, tenant, project_id)`. Without it, resolving vector search results via `get_memories_by_embedding_ids` resulted in an O(N) full table scan.
**Action:** Add the composite index `(embedding_id, tenant, project_id)` to all memory tables to optimize resolution of search results.

## 2024-05-14 - Missing composite index for vector search resolution
**Learning:** The memory tables (`memories_semantic`, `memories_episodic`, `memories_procedural`) lack a composite index on `(embedding_id, tenant, project_id)`. Without it, resolving vector search results via `get_memories_by_embedding_ids` causes O(N) full table scans since `embedding_id` checks with multiple tenant/project conditions are slow.
**Action:** Always ensure foreign key equivalents used in bulk queries (e.g. `embedding_id IN (...) AND tenant = ? AND project_id = ?`) are appropriately indexed.

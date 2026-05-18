## 2024-05-18 - Composite index for vector search results
**Learning:** Resolving vector search results via `get_memories_by_embedding_ids` causes O(N) full table scans without an index.
**Action:** Always ensure memory tables (`memories_semantic`, `memories_episodic`, `memories_procedural`) have a composite index on `(embedding_id, tenant, project_id)`.

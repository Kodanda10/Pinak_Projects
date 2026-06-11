## 2024-06-11 - Optimize memory table indexes
**Learning:** The memory tables (`memories_semantic`, `memories_episodic`, `memories_procedural`) utilize a composite index on `(embedding_id, tenant, project_id)` to prevent O(N) full table scans during vector search, but this composite index is missing in the database schema.
**Action:** Always verify if indices mentioned in codebase knowledge/documentation actually exist in the database schema.

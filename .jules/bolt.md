## 2026-07-07 - Add missing composite indexes for vector search
**Learning:** The method `get_memories_by_embedding_ids` retrieves memory entries using an `IN` clause over `embedding_id` along with `tenant` and `project_id` filters. Without a composite index, SQLite defaults to an O(N) full table scan for every vector query, which is a significant bottleneck.
**Action:** Add composite indices on `(embedding_id, tenant, project_id)` to all memory layers (`memories_semantic`, `memories_episodic`, `memories_procedural`) to ensure efficient covering index lookups during vector search.

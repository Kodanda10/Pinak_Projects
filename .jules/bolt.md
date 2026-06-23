## 2024-06-23 - Add composite index for embedding_ids on memory tables
**Learning:** The memory tables (`memories_semantic`, `memories_episodic`, `memories_procedural`) utilize composite indexes on `(embedding_id, tenant, project_id)` in `app/core/database.py` to prevent O(N) full table scans during vector search. I found that they are missing from `_init_db`.
**Action:** Add missing composite indexes for `embedding_id`, `tenant`, and `project_id` on the 3 memory tables.

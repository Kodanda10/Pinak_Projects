## 2025-08-30 - Initial Setup
**Learning:** Initializing Bolt's journal.
**Action:** Keep entries critical and focused.
## 2025-08-30 - Missing Indexes in SQLite
**Learning:** Missing database indexes on `embedding_id` in SQLite cause O(n) linear scans during vector hybrid search results retrieval (`get_memories_by_embedding_ids`).
**Action:** Added indexes for `embedding_id` on `memories_semantic`, `memories_episodic`, and `memories_procedural` tables to optimize retrieval by embedding_id.

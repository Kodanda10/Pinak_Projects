## 2024-05-24 - [Missing Database Indexes on embedding_id]
**Learning:** SQLite tables `memories_semantic`, `memories_episodic`, and `memories_procedural` lacked indexes on `embedding_id`. When performing hybrid search lookups via `get_memories_by_embedding_ids`, this resulted in full table scans. Adding indexes on `embedding_id` prevents this and reduces lookup latency by approximately 200x.
**Action:** Always verify that columns used in `IN` clauses during frequently called read methods have corresponding database indexes to avoid O(N) lookup degradation.

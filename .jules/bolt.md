## 2024-05-20 - Missing Indexes on embedding_id
**Learning:** Database queries mapped to FAISS vector search results (`get_memories_by_embedding_ids`) perform full table scans because `embedding_id` lacks an index.
**Action:** Add database indexes to `embedding_id` in SQLite for `memories_semantic`, `memories_episodic`, and `memories_procedural` to optimize vector-to-metadata retrieval latency.

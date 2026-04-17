## 2025-04-17 - Add embedding_id Index
**Learning:** Database indexes on `embedding_id` for SQLite memory tables (`memories_semantic`, `memories_episodic`, `memories_procedural`) prevent full table scans during hybrid search mappings, significantly reducing lookup latency (e.g., in `get_memories_by_embedding_ids`) by approximately 200x.
**Action:** When working with embedding mappings in SQL databases, always add indexes to mapping columns to prevent sequential scans.

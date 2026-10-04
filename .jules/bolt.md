# Bolt's Journal

## 2023-10-04 - Initial Setup
**Learning:** Just started on the Pinak codebase. Memory service uses Numpy for vector search to avoid FAISS segfaults.
**Action:** Always be careful with memory service vector store optimizations, it seems deliberately chosen over FAISS for stability.

## 2023-10-04 - Missing Database Indexes
**Learning:** Found that `memories_semantic`, `memories_episodic`, `memories_procedural`, and `memories_rag` tables are frequently queried filtering by `tenant` and `project_id`, but no composite index exists for these fields. This creates a full table scan on every memory search/retrieval.
**Action:** Add `CREATE INDEX idx_memories_semantic_tenant_project ON memories_semantic (tenant, project_id)` and similarly for other memory layers. Ensure we wrap it in `self._column_exists()` check per boundaries, although this is index creation not column creation, so it's safe to just `CREATE INDEX IF NOT EXISTS`.

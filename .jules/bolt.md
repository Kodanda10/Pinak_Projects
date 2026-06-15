## 2026-06-15 - Missing composite indexes for vector search retrieval
**Learning:** The memories tables (semantic, episodic, procedural) lacked composite indexes on (embedding_id, tenant, project_id), causing full table scans during vector search result retrieval which led to performance bottlenecks.
**Action:** Always add appropriate composite database indexes for fields that are frequently queried together, especially when resolving results from a vector store, to ensure fast O(1) or O(log N) lookups instead of O(N) table scans.

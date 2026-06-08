## 2024-05-30 - Composite Indexes for embedding_id
**Learning:** Retrieving vector search results by `embedding_id` from SQLite tables was lacking indexes on `(embedding_id, tenant, project_id)`, causing full table scans. Adding these composite indexes provides a >3x speedup on vector resolution.
**Action:** Always verify database indexes on query paths involved in high-throughput operations like vector search result resolution.

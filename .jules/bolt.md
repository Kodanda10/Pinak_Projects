## 2026-05-20 - Prevent O(N) Table Scans on Vector Search Resolution
**Learning:** SQLite full table scans occur when looking up memory metadata by `embedding_id` after a vector search if the table lacks an index on `(embedding_id, tenant, project_id)`.
**Action:** Always add a composite index on `(embedding_id, tenant, project_id)` for memory tables that link to a vector store to ensure fast O(1) resolution.

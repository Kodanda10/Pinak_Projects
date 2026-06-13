## 2024-06-13 - Add composite database indexes for vector search and counts
**Learning:** SQLite composite indexing is critical to prevent O(N) table scans during vector search and count operations in multi-tenant environments.
**Action:** Add composite indexes on `(embedding_id, tenant, project_id)` for vector searches and `(client_id, tenant, project_id, status)` for count queries.

## 2025-08-29 - Missing Composite Indices in SQLite
**Learning:** The memory tables (`memories_semantic`, `memories_episodic`, `memories_procedural`) use O(N) full table scans during vector search because composite indexes on `(embedding_id, tenant, project_id)` are missing. Also `logs_client_issues` and `memory_quarantine` lack composite indices for frequent multi-tenant lookups.
**Action:** Add composite indices for `(embedding_id, tenant, project_id)` and `(client_id, tenant, project_id, status)` to prevent full table scans and improve performance.

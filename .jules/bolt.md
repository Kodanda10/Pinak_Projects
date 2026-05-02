
## 2024-05-02 - SQLite Missing Indexes for get_client_layer_stats
**Learning:** The Pinak memory service's SQLite database lacked composite indexes on `(tenant, project_id, client_id)` for its 5 core memory tables, causing heavy full table scans whenever `get_client_layer_stats` aggregated multi-tenant data.
**Action:** Created composite indexes (`idx_{table}_tenant_project_client`) at the end of `_init_db` (after all `_ensure_column` migrations have run) for `memories_semantic`, `memories_episodic`, `memories_procedural`, `memories_rag`, and `working_memory` to optimize aggregate queries and prevent full scans.

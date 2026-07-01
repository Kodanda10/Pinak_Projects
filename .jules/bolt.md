## 2026-07-01 - Optimize Memory Vector Search
**Learning:** The memories tables (semantic, episodic, procedural) lacked composite indexes on (embedding_id, tenant, project_id). Since get_memories_by_embedding_ids queries by embedding_id IN (...) combined with tenant/project constraints, missing indices triggered full table scans per table during vector retrieval.
**Action:** Always verify query plans for IN clauses on frequently queried fields, and use composite indices when multiple exact-match constraints (tenant, project) are present.

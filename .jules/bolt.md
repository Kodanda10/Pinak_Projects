## 2025-08-29 - Missing `embedding_id` in `memories_rag`
**Learning:** Unlike other memory tables, the `memories_rag` table in `app/core/database.py` does not contain an `embedding_id` column.
**Action:** When creating composite indexes for multi-tenant queries on the `memories_rag` table, use `(tenant, project_id)` instead of `(tenant, project_id, embedding_id)`.

## 2024-05-24 - Missing SQLite Indexes for Multi-Tenant Vector Retrieval
**Learning:** Vector search uses FAISS which returns a list of embedding_ids. The SQL retrieval query then uses `embedding_id IN (...)` along with tenant and project filtering. Without an index on `(tenant, project_id, embedding_id)`, SQLite performs O(N) full table scans for each batch of results, crippling retrieval performance on large memory datasets.
**Action:** Always create composite indexes `(tenant, project_id, embedding_id)` for memory tables to ensure O(1) multi-tenant retrieval.

## 2025-07-21 - Missing Index on Multi-tenant Architecture
**Learning:** The database heavily queries memories filtering by tenant and project_id, but lacks composite indices for these fields, resulting in full table scans for every memory retrieval query (N+1/SCAN problem in multi-tenant environments).
**Action:** Create composite indices on (tenant, project_id, embedding_id) for the memory tables and (tenant, project_id) for other frequently queried tables to avoid full table scans.

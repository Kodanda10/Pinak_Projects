## 2024-06-27 - Composite Indices for Multi-Tenant Vector Search
**Learning:** SQLite queries with multi-tenant filters (`embedding_id IN (...)`, `tenant = ?`, `project_id = ?`) fall back to O(N) full table scans if no index covering all queried columns exists.
**Action:** Always add composite indexes on `(embedding_id, tenant, project_id)` to prevent O(N) full table scans during vector search.

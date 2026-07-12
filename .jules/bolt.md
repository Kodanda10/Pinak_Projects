## 2024-07-12 - Multi-tenant filter indexing in SQLite
**Learning:** In multi-tenant environments filtering by (client_id, tenant, project_id, status), a single-column index on 'status' causes inefficient linear scans on the remaining fields. A composite index enables covering index usage, improving query performance.
**Action:** Use composite or covering indexes incorporating all filter fields when optimizing multi-tenant SQLite queries.

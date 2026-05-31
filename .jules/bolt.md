## 2026-05-31 - SQLite Composite Index Optimization for Count Queries
**Learning:** In SQLite, adding composite indexes matching the exact query predicates (e.g., `(client_id, tenant, project_id, status)`) prevents expensive O(N) full table scans and significantly speeds up `COUNT(*)` aggregation queries.
**Action:** Always verify if high-frequency filtering and aggregation queries have matching composite indexes, especially for multi-tenant tables.

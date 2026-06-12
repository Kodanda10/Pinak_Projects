## 2024-02-27 - Database Composite Indexes Optimization
**Learning:** In SQLite, adding a composite index containing fields (e.g., client_id, tenant, project_id, status) changes an O(N) full table scan to an O(log N) lookup for count and search queries. This drastically improves backend performance when tables like `logs_client_issues` or `memories_semantic` grow large.
**Action:** Add composite indices (via try-except blocks to ignore operational errors on backwards compatibility) for heavily-queried sets of columns in SQLite databases.

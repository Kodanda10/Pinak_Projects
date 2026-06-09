## 2024-05-24 - Composite Indexes for Count Queries
**Learning:** The memory service architecture frequently counts issues and quarantine records filtering by `(client_id, tenant, project_id, status)`. Without composite indexes on these columns, the database performs O(N) full table scans which creates a bottleneck.
**Action:** Always ensure composite indexes are added at the end of `DatabaseManager._init_db()` (wrapped in a try-except to avoid breaking legacy schemas) for columns frequently queried together in counts.

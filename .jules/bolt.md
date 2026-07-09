## 2026-07-09 - Multi-tenant SQLite Indexing
**Learning:** Single-column indexes on highly selective fields (like status) cause inefficient linear scans for multi-tenant queries that also filter on client_id, tenant, and project_id.
**Action:** Always use composite or covering indexes that incorporate all these multi-tenant filter fields to enable fast covering index usage and prevent O(N) full table scans.

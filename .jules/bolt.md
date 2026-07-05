## 2026-07-05 - SQLite Composite Index Optimization
**Learning:** When optimizing multi-tenant queries in SQLite that filter by multiple fields (like `client_id`, `tenant`, `project_id`, and `status` or `embedding_id`), single-column indexes cause inefficient linear scans for the remaining filters.
**Action:** Always use composite or covering indexes incorporating all filtered fields to prevent O(N) full table scans.

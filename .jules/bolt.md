## 2024-06-06 - Missing documented indexes in SQLite schema
**Learning:** The documentation claimed composite indexes on `(embedding_id, tenant, project_id)` existed to prevent O(N) full table scans during vector search, but they were missing from the database initialization.
**Action:** Always verify documented database schema optimizations against the actual database initialization code.

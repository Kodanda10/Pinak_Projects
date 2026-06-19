## 2025-08-29 - Missing composite index causes O(N) table scans during vector search
**Learning:** The memory tables utilize vector searches to fetch top-K IDs and then querying the database filtering by `IN (embedding_ids)` as well as tenant/project_id. Without a composite index, this causes a full table scan on every memory search.
**Action:** Always verify query plans for IN clauses especially when combined with multi-tenant filtering. Ensure composite indices on `(embedding_id, tenant, project_id)` exist.

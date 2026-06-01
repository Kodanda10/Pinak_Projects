## 2026-06-01 - Fix SQL injection risk in dynamic UPDATE query
**Vulnerability:** Medium severity SQL injection risk via dynamically building SET clauses in update_memory.
**Learning:** The method used arbitrary dictionary keys directly in the SQL string, potentially allowing column name injection.
**Prevention:** Enforce that dictionary keys are valid strings and Python identifiers using `if not isinstance(key, str) or not key.isidentifier():` before constructing the query.

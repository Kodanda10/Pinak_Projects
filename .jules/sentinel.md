## 2024-05-20 - SQL Injection via Dictionary Keys in Dynamic Update
**Vulnerability:** SQL Injection in `update_memory` caused by dynamically generating `SET` clauses from dictionary keys (`updates.keys()`) without validating that they are valid column names.
**Learning:** Even when using parameterized queries for values (e.g., `SET col = ?`), trusting dictionary keys to construct the SQL structure directly allows attackers to inject arbitrary SQL or modify columns they shouldn't (like `created_at` or `tenant` - IDOR).
**Prevention:** Always strictly validate column names using an allowlist or `isidentifier()`, and explicitly remove protected fields (like `id`, `tenant`, `project_id`) from any dynamic update payloads.

## 2026-05-31 - Mitigate dynamic SQL injection on dictionary keys
**Vulnerability:** SQL injection vulnerability in `update_memory` due to unsanitized dictionary keys used to construct dynamic `UPDATE` query (`SET {set_clause}`).
**Learning:** Dictionary keys used for dynamic column names must be validated to prevent attackers from injecting arbitrary SQL.
**Prevention:** Explicitly enforce that dictionary keys are valid strings and Python identifiers using `if not isinstance(key, str) or not key.isidentifier(): raise ValueError(...)` before constructing the query.

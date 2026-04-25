## 2026-04-25 - Prevent SQL Injection via Dynamic Column Names
**Vulnerability:** SQL Injection in `DatabaseManager.update_memory` due to unsanitized dictionary keys used as column names in dynamic `UPDATE` queries.
**Learning:** Python DB API parameterization (e.g., `execute("...", params)`) only protects values passed to placeholders (`?`), not identifiers like column names. Dynamically constructing SQL using user-provided dictionary keys is an injection vector.
**Prevention:** Explicitly enforce that dictionary keys are valid strings and Python identifiers using `if not isinstance(key, str) or not key.isidentifier(): raise ValueError(...)` before constructing dynamic queries.

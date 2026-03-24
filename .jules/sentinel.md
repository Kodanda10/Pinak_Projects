## 2024-05-24 - SQL Injection Vulnerability in Update Memory
**Vulnerability:** SQL Injection in `DatabaseManager.update_memory` due to unsanitized dictionary keys used to construct SET clauses for SQL queries.
**Learning:** Python DB API parameterization (e.g., `execute("...", params)`) only protects values passed to placeholders (`?`), not identifiers like column names. Dynamically constructing SQL using user-provided dictionary keys is an injection vector and must be sanitized.
**Prevention:** Ensure that dynamically constructed SQL identifiers (like column names) are validated using strict whitelists or logic like `isinstance(key, str)` and `key.isidentifier()`, and ideally checked against the actual schema, to prevent unauthorized SQL execution.

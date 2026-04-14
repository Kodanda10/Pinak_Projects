## 2026-04-14 - SQL Injection via Dictionary Keys as Column Names
**Vulnerability:** In `DatabaseManager.update_memory`, user-provided dictionary keys were being dynamically used to construct the SQL `SET` clause (`f"{k} = ?"`). This is a SQL injection vulnerability because DB API parameterization only protects values, not identifiers like column names.
**Learning:** Even when using parameterization for values, constructing SQL strings with untrusted keys (e.g. from JSON payloads) allows attackers to inject arbitrary SQL logic or modify unintended columns.
**Prevention:** Always sanitize dynamically constructed SQL identifiers. In this case, ensure keys are valid Python identifiers using `isinstance(key, str) and key.isidentifier()` before appending them to the query string.

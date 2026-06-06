## 2026-06-06 - SQL Injection via Dictionary Keys in Dynamic SQL

**Vulnerability:** Unsanitized dictionary keys were being directly interpolated into dynamic SQL query strings in `DatabaseManager.update_memory()`. While values were properly parameterized, an attacker controlling the JSON payload keys could potentially inject arbitrary SQL fragments into the `UPDATE` statement's `SET` clause.
**Learning:** Python's f-strings or `.join()` used for dynamic column names are a common SQL injection vector if the input keys are untrusted.
**Prevention:** Always explicitly validate that dictionary keys used in dynamic SQL construction are valid identifiers (e.g., `isinstance(key, str)` and `key.isidentifier()`) and ideally explicitly belong to an allowlist of valid column names.

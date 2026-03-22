## 2024-05-24 - SQL Injection via Dictionary Keys in update_memory
**Vulnerability:** In `DatabaseManager.update_memory`, dictionary keys are concatenated directly into the SQL UPDATE query without sanitization. This allows SQL injection if arbitrary keys are passed.
**Learning:** Python DB API parameterization only protects values passed to placeholders, not identifiers like column names. Dynamically constructing SQL using user-provided dictionary keys is an injection vector.
**Prevention:** Keys must be sanitized, e.g. using `isinstance(key, str) and key.isidentifier()`, before being interpolated into SQL queries.

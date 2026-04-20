## 2024-05-24 - SQL Injection in DatabaseManager.update_memory via dynamic dict keys
**Vulnerability:** The `update_memory` method constructs an `UPDATE` SQL query using dictionary keys from the `updates` parameter dynamically via f-strings (`set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])`). This allows SQL injection if malicious keys are passed.
**Learning:** Python DB API parameterization protects values (passed as `?`), but not identifiers like column names. Dynamically constructing SQL with user-provided keys is an injection vector.
**Prevention:** Validate dictionary keys directly using `isinstance(key, str) and key.isidentifier()` to ensure only safe column names are interpolated into the SQL string.

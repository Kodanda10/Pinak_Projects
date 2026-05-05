## 2025-05-05 - Dynamic Key Validation in UPDATE Statements
**Vulnerability:** SQL Injection/Mass Assignment via unvalidated dictionary keys used as column names in `DatabaseManager.update_memory`.
**Learning:** The method constructed an `UPDATE` query by iterating through the keys of the `updates` dictionary and joining them as `{k} = ?` without validating that `k` is a legitimate column name. This allows injection of SQL fragments if an attacker controls the dictionary keys.
**Prevention:** Always validate that dynamic dictionary keys intended to be used as column names are valid Python identifiers using `isinstance(key, str)` and `key.isidentifier()`.

## 2024-04-28 - Dynamic Column Names in SQL `UPDATE` Lead to SQL Injection
**Vulnerability:** A SQL injection vector was present in `DatabaseManager.update_memory` where user-provided dictionary keys were directly used to construct the `SET` clause of an `UPDATE` query without adequate sanitization or verification that they represent valid table columns.
**Learning:** Python's DB API parameterization (using `?`) only protects the values bound to placeholders, not the identifiers like column names used directly in the `f-string` formatted query.
**Prevention:** Explicitly check dictionary keys representing column names with `isinstance(key, str) and key.isidentifier()` to ensure they are valid Python/SQL identifiers before injecting them into a dynamically generated query.

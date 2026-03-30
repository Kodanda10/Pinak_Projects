## 2024-05-01 - SQL Injection in Dynamic UPDATE Statements via Dictionary Keys
**Vulnerability:** Dictionary keys in user-provided input were directly used to construct SQL SET clauses via string interpolation in `DatabaseManager.update_memory`, creating a SQL injection risk.
**Learning:** Python DB API parameterization only protects values (`?`), not identifiers. Dynamically constructing SQL using user-provided dictionary keys is an injection vector and must be sanitized.
**Prevention:** Validate all dictionary keys that are used as SQL identifiers (e.g., column names) using strict checks such as `isinstance(key, str) and key.isidentifier()` to ensure they are safe, instead of trying to escape them or relying on parameterization.

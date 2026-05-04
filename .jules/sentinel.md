## 2025-02-28 - Dynamic Query Column Validation
**Vulnerability:** SQL Injection via dynamically constructed UPDATE statement SET clauses.
**Learning:** `DatabaseManager.update_memory` relied on user-controlled dictionary keys to format the `SET` clauses. While parameterized for values, the column names (keys) were not validated. Linter tools like Bandit flag this (`B608`).
**Prevention:** Always validate that dynamic dictionary keys intended for column names in SQL queries are valid strings and identifiers using `isinstance(key, str)` and `key.isidentifier()`. Only append the Bandit `#nosec` directive once this active validation is strictly in place.

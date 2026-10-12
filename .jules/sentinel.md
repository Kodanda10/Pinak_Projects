## 2024-10-12 - Prevent SQL Injection via Unvalidated Column Names
**Vulnerability:** Found a SQL injection vulnerability in `DatabaseManager.update_memory` where dynamic dictionary keys from `updates` were concatenated directly into the `UPDATE` statement's `SET` clause (`set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])`).
**Learning:** Dynamic column names from user input must be rigorously validated. Parameterized queries (`?`) only protect the values, not the column identifiers.
**Prevention:** Use Python's `.isidentifier()` method on dynamic column keys before inclusion in SQL, and maintain an explicit set of protected fields (`id`, `tenant`, `project_id`, `created_at`) that cannot be modified, enforcing both security and multi-tenant isolation.

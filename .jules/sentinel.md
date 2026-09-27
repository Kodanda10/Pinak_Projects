## 2026-09-27 - SQL Injection in Dynamic Column Names
**Vulnerability:** SQL injection vulnerability via dynamically generated column names from user input in `DatabaseManager.update_memory()`. Keys of the `updates` dictionary were directly interpolated into the `SET` clause of the SQL `UPDATE` statement without validation.
**Learning:** Even when using parameter binding for values, directly injecting dictionary keys as column names in SQL queries opens up the possibility for SQL injection if the keys are not validated.
**Prevention:** Validate all dynamically generated column names (e.g., using `.isidentifier()`) and explicitly prevent modification of protected fields (like `id`, `tenant`, and `project_id`) to avoid IDOR vulnerabilities.

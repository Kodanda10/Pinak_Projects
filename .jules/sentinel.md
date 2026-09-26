
## 2024-05-24 - Dynamic SQL Injection in SQLite Interpolation
**Vulnerability:** A critical SQL injection vulnerability existed in `DatabaseManager.update_memory` where unvalidated dictionary keys were directly interpolated into a `SET` clause via `f"{k} = ?"`.
**Learning:** Even when using parameterized queries for values (e.g., `?`), dynamically constructing column names from user input without strict validation creates an injection vector, allowing attackers to overwrite protected columns or modify the query structure.
**Prevention:** Always validate dynamic column names against a strict allowlist or use Python's `.isidentifier()` method to ensure they only contain safe characters, and explicitly block modifications to protected, system-controlled columns (like `id`, `tenant`, and `project_id`).

## 2025-02-14 - Fix SQL Injection in DatabaseManager.update_memory
**Vulnerability:** SQL injection vulnerability in `Pinak_Services/memory_service/app/core/database.py` where user-provided keys in `updates` dictionary were concatenated into SET clauses without validation.
**Learning:** Dynamically generating SQL statements based on user input without strictly validating column names opens the door to SQL injection, even if parameter binding is used for values.
**Prevention:** Use `.isidentifier()` to validate that dynamically generated column names are safe, and explicitly prevent updates to protected fields (e.g., `id`, `tenant`, `project_id`).

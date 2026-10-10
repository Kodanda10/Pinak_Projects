## 2024-10-10 - Fix SQL Injection and IDOR in update_memory
**Vulnerability:** The `update_memory` function dynamically built SQL `SET` clauses from user-supplied keys without validation, allowing SQL injection. It also allowed modification of protected fields (`id`, `tenant`, `project_id`).
**Learning:** Even when using parameterized queries for values, using unsanitized user input for column names exposes the system to SQL injection and IDOR.
**Prevention:** Always validate dynamically generated column names using `.isidentifier()` and explicitly exclude protected fields from updates.

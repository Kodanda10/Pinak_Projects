## 2023-10-25 - Dynamic Column Name SQL Injection
**Vulnerability:** In `update_memory` within `DatabaseManager`, dictionary keys were interpolated directly into a SQL `UPDATE` statement string (`f"{k} = ?"`). An attacker could bypass the `WHERE` clause or overwrite unauthorized columns (like `tenant` or `project_id`) by injecting SQL via a malformed JSON key (e.g., `{"tenant = 'attacker', value": "hacked"}`).
**Learning:** Even when parameterized queries are used for values, dynamically interpolating column names without strict validation opens the door for SQL injection.
**Prevention:** Always validate dynamically generated SQL components using strict allowlists or structural checks like `str.isidentifier()`, and explicitly discard protected fields.

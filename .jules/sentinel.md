## 2024-11-06 - [Dynamic Column Generation Mass Assignment]
**Vulnerability:** The `update_memory` function iterated over all keys in the input JSON and dynamically constructed an SQL `UPDATE` statement. This allowed updating protected fields (like `tenant`, `project_id`) or injecting non-existent columns.
**Learning:** In dynamically constructed SQL, we cannot trust that the JSON keys correspond only to valid, modifiable schema columns.
**Prevention:** Always validate all dynamic JSON keys against a strict hardcoded whitelist of allowed updatable columns before concatenating them into an SQL query.

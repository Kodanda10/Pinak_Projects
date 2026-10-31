## 2024-05-24 - Fix Mass Assignment and SQL Injection in update_memory
**Vulnerability:** The `update_memory` function dynamically generated SQL `SET` clauses from arbitrary user input without any sanitization or whitelisting, allowing both SQL injection via malicious dictionary keys and mass assignment (e.g. overwriting protected fields like `tenant`, `id`, `project_id`).
**Learning:** Building dynamic SQL from dictionaries is inherently dangerous. Unsanitized dictionary keys used to generate column names lead directly to vulnerabilities.
**Prevention:** Always use strict whitelists of allowed columns when building dynamic SQL queries and validate keys using `.isidentifier()`. Protected fields should never be updatable.

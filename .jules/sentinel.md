## 2024-05-23 - Prevent Mass Assignment and SQL Injection in update_memory
**Vulnerability:** Dynamic column generation in SQL UPDATE statements (`update_memory`) relied on unvalidated keys from user-provided dictionaries, allowing SQL injection and unauthorized updates to protected fields (Mass Assignment).
**Learning:** The previous implementation directly formatted dictionary keys into the `SET` clause (`f"{k} = ?"`) without verifying if the keys matched known, safe column names or filtering out protected attributes like `tenant` and `project_id`.
**Prevention:** Always validate dynamically generated SQL columns against a strict whitelist of allowed identifiers and use Python's `.isidentifier()` method to ensure safe formatting.

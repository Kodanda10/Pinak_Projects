## 2024-05-18 - Mass Assignment/SQL Injection in update_memory
**Vulnerability:** The `update_memory` function in `DatabaseManager` constructed SQL `SET` clauses directly from the keys of the `updates` dictionary without validating if the keys were actually valid columns. This allowed mass assignment (updating system fields) and SQL injection (e.g. `{"content = 'Hacked' --": "ignored"}`).
**Learning:** Even if values are parameterized, dictionary keys used to construct column names in SQL queries must be strictly validated or whitelisted to prevent SQL injection and mass assignment.
**Prevention:** Always use `.isidentifier()` and a strict whitelist of allowed columns when dynamically generating SQL from dictionaries.

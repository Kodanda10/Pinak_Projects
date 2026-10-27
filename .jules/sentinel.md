## 2025-02-14 - Mass Assignment in Database update_memory
**Vulnerability:** Found a mass assignment and dynamic column generation vulnerability in `update_memory` of `DatabaseManager` where arbitrary dictionary keys were concatenated directly into the SQL `UPDATE` statement.
**Learning:** This existed because the `updates` dictionary was mapped blindly to table columns without validation, allowing clients to inject unrecognized or protected column modifications and potentially cause SQL syntax errors.
**Prevention:** Implement an explicit whitelist of allowed modifiable columns and validate each key with `.isidentifier()` before inserting it into a dynamic query generation clause.

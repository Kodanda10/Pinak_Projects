## 2024-05-18 - [CRITICAL] SQL Injection in update_memory
**Vulnerability:** SQL Injection in `DatabaseManager.update_memory` where unsanitized keys from the `updates` dictionary are directly formatted into the SET clause of an UPDATE query (`f"{k} = ?"`).
**Learning:** Even though the application logic (`memory_service.py`) filters some keys using `forbidden_keys`, the database layer allows arbitrary dictionary keys. This allows attackers to inject arbitrary SQL logic into column names (e.g., `content = ?, other_col = (SELECT 1) --`).
**Prevention:** Always validate dictionary keys used in dynamic SQL queries using `str.isidentifier()` to ensure they are valid SQL identifiers, and avoid trusting application-level key filters to protect the database layer.

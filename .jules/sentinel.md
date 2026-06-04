## 2026-06-04 - Fix SQL Injection in update_memory
**Vulnerability:** SQL injection and mass assignment via unsanitized dictionary keys in dynamic SQL query construction within `DatabaseManager.update_memory`.
**Learning:** Hardcoded string concatenation for SQL queries using user-provided dictionary keys allowed potential SQL injection and unauthorized column modification (mass assignment). This existed because the codebase assumed keys were safe attribute names.
**Prevention:** Always validate that dictionary keys used in dynamic SQL construction are valid strings and Python identifiers using `isinstance(key, str) and key.isidentifier()`.

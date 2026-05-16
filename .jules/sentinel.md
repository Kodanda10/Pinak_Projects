## 2026-05-16 - Prevent SQL Injection via dynamic dictionary keys
**Vulnerability:** SQL Injection in DatabaseManager.update_memory via unsanitized dictionary keys used to construct SET clause.
**Learning:** Dictionary keys injected dynamically into SQL strings can bypass ORM/driver protections if not properly validated.
**Prevention:** Explicitly enforce that dynamic dictionary keys used in SQL strings are strings and valid Python identifiers using isinstance(key, str) and key.isidentifier().

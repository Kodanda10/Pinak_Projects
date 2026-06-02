## 2026-06-02 - SQL Injection in Dynamic UPDATE Queries
**Vulnerability:** SQL injection and mass assignment vulnerability in `DatabaseManager.update_memory` due to unsanitized dictionary keys used to construct the `SET` clause in `UPDATE` statements.
**Learning:** Dynamic query builders must never blindly trust dictionary keys, as an attacker could supply malicious keys like `"is_admin = 1 --"` to bypass parameterization.
**Prevention:** Explicitly validate that all dictionary keys used in dynamic SQL construction are valid strings and Python identifiers using `if not isinstance(key, str) or not key.isidentifier(): raise ValueError(...)`.

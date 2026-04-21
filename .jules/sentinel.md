## 2025-02-27 - [Dynamic SQL Key Construction]
**Vulnerability:** SQL Injection in `DatabaseManager.update_memory` where user-provided dictionary keys were directly formatted into the `SET` clause of an `UPDATE` query without validation.
**Learning:** Python's DB API `?` placeholders only parameterize values, not identifiers (like column names). While Bandit flagged the f-string, the real vulnerability was the lack of key sanitization in dynamic query building.
**Prevention:** Always validate dynamically constructed SQL identifiers (e.g., column names) against an allowlist or using strict format checks like `.isidentifier()` before inserting them into query strings.

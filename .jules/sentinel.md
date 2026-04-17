## 2024-04-17 - [SQL Injection via Dynamic Column Updates]
**Vulnerability:** The `update_memory` method in `DatabaseManager` dynamically constructed SQL `UPDATE` statement `SET` clauses using user-provided dictionary keys. Because standard DB-API parameterization (`?`) only protects values, malicious keys could inject arbitrary SQL structure.
**Learning:** Python dictionary keys used as dynamic identifiers are not protected by parameterized queries and represent a critical injection vector when mass-updating rows if not validated.
**Prevention:** Strictly validate dynamic dictionary keys that translate into SQL identifiers (e.g. column names) using `isidentifier()` to ensure they contain only valid alphanumeric and underscore characters, preventing SQL injection strings.

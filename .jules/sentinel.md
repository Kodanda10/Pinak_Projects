## 2025-08-29 - [CRITICAL] SQL Injection in update_memory
**Vulnerability:** SQL Injection in `DatabaseManager.update_memory` via unchecked dict keys used in `SET` clause concatenation.
**Learning:** Keys of dynamic `updates` dictionaries must be validated as safe SQL identifiers (`str.isidentifier()`) when constructing dynamic SQL queries, as attackers can control these keys to bypass authorization or inject logic.
**Prevention:** Always validate dictionary keys using `str.isidentifier()` before interpolating them into a dynamic SQL query, or restrict the allowed keys to a strict allowlist.

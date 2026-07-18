## 2025-02-12 - Prevent SQL Injection in Dynamic Queries
**Vulnerability:** Dynamic SQL `UPDATE` statements in `DatabaseManager.update_memory` were constructed using unvalidated dictionary keys (`set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])`), allowing SQL injection if untrusted keys are passed.
**Learning:** Even when parameterized values (`?`) are used, dynamic column names generated from user-supplied dictionaries must be strictly validated against an allowlist or identifier constraints to prevent injection via structural manipulation.
**Prevention:** Always validate dynamic column names using `str.isidentifier()` or an explicit allowlist before constructing SQL clauses.

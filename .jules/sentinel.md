## 2026-05-21 - [Mass Assignment / SQL Injection in Dynamic Queries]
**Vulnerability:** Dynamic SQL updates in `app/core/database.py` trusted dictionary keys directly without validation.
**Learning:** When building `set_clause` for dynamic SQL updates, dictionary keys must be explicitly validated as strings and valid identifiers to prevent injection or mass assignment vulnerabilities.
**Prevention:** Explicitly enforce `if not isinstance(key, str) or not key.isidentifier(): raise ValueError(...)` before constructing dynamic queries.

## 2024-04-11 - [Title]
**Vulnerability:** Found SQL injection in database update queries where user-provided dictionaries are unpacked directly into `SET` clauses.
**Learning:** Python parameterization only protects values, not identifiers. Dynamically constructing SQL using user-provided keys is an injection vector and must be sanitized.
**Prevention:** Always validate that dynamic SQL identifier strings are valid strings and Python identifiers (`isidentifier()`) before incorporating them into the SQL structure.

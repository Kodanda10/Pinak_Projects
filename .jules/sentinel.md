## 2025-05-26 - Prevent SQL Injection via Dynamic Column Names
**Vulnerability:** Constructing dynamic UPDATE statements using f-strings based on dictionary keys (e.g. `f"UPDATE {table} SET {set_clause} ..."`) without strong validation allows SQL injection.
**Learning:** Relying on upstream dictionary sanitization is insufficient and violates defense-in-depth principles. Attackers can inject SQL into the column structure if `set_clause` is constructed unsafely.
**Prevention:** To safely build dynamic SQL queries, always enforce that dictionary keys are valid strings and Python identifiers using `if not isinstance(key, str) or not key.isidentifier(): raise ValueError(...)` before constructing the query.

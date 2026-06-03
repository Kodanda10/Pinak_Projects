## 2024-06-03 - Dynamic SQL Construction with Dictionary Keys
**Vulnerability:** SQL injection and mass assignment via unvalidated dictionary keys used in dynamic `UPDATE` queries (e.g. `set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])`).
**Learning:** Hardcoded queries with parameters are safe, but dynamically constructing the query string using keys from user-provided dictionaries can still lead to SQLi if those keys are not strictly validated.
**Prevention:** To mitigate SQL injection and mass assignment when building dynamic SQL queries, explicitly enforce that dictionary keys are valid strings and Python identifiers using `if not isinstance(key, str) or not key.isidentifier(): raise ValueError(...)` before constructing the query.

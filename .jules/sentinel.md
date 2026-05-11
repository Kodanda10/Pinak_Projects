## 2026-05-11 - [Mitigate SQL injection in DatabaseManager]
**Vulnerability:** SQL injection via string-based query construction when updating dictionary fields. The keys of `updates` dict were directly interpolated into an `UPDATE` query's `SET` clause without validation (`set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])`).
**Learning:** Keys in dynamic update dicts must be strictly validated as identifiers before interpolation to prevent mass assignment/SQLi via crafted keys.
**Prevention:** Use `if not isinstance(key, str) or not key.isidentifier(): raise ValueError(...)` to assert keys are legitimate table columns.

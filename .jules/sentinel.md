## 2026-04-30 - Prevent SQL Injection via dynamic dictionary keys
**Vulnerability:** SQL Injection in `DatabaseManager.update_memory` due to dynamic column names constructed from unvalidated dictionary keys in a JSON payload.
**Learning:** Using f-strings to build SQL queries based on dictionary keys (e.g. `set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])`) allows attackers to inject malicious SQL logic since parameterization only protects values, not column names.
**Prevention:** Always strictly validate dynamic keys before constructing the SQL statement, ideally using `isinstance(key, str) and key.isidentifier()` to ensure they are valid alphanumeric identifiers, raising a `ValueError` if they are not.

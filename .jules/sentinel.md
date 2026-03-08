## 2024-05-24 - [CRITICAL] SQL Injection and Mass Assignment in Memory Update

**Vulnerability:**
The `DatabaseManager.update_memory` function was constructing dynamic SQL SET clauses directly from dictionary keys (`set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])`). This allowed for SQL injection via dictionary keys. Furthermore, `MemoryService.update_memory` was only stripping out a small set of "forbidden" system fields, leaving the service vulnerable to mass assignment attacks.

**Learning:**
Any dynamically constructed SQL query relying on externally provided dictionary keys must validate the keys as well as the values. Simple key exclusion logic (`forbidden_keys`) is insufficient for a data service with multiple layers, as an attacker could inject attributes belonging to other layers, bypass checks, or inject SQL code.

**Prevention:**
Enforce strict identifier validation (`isinstance(key, str) and key.isidentifier()`) when constructing SQL queries from dictionary keys. Use an explicit, layer-specific `ALLOWED_UPDATES` whitelist instead of an exclusion list to prevent mass assignment vulnerabilities.

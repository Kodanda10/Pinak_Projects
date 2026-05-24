## 2026-05-24 - Prevent SQL Injection via Dynamic Dictionary Keys
**Vulnerability:** A SQL injection vulnerability existed in `update_memory` due to directly interpolating dictionary keys into the SET clause without validation.
**Learning:** When building dynamic SQL queries based on user-provided dictionaries, keys must be strictly validated as safe identifiers (e.g., using `isidentifier()`) because parameterized queries only protect values, not column names.
**Prevention:** Enforce `if not isinstance(key, str) or not key.isidentifier(): raise ValueError(...)` for any dynamically constructed SQL schema elements.

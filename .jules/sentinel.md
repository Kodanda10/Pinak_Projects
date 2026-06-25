## 2025-06-25 - Prevent SQL Injection via Dictionary Keys in Dynamic Queries
**Vulnerability:** SQL injection vector found in `update_memory` where unsanitized keys from the `updates` dictionary were interpolated directly into the `SET` clause of the SQL statement (`set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])`).
**Learning:** Application-level key filters (like `forbidden_keys`) do not guarantee safety if the database layer dynamically constructs queries from dictionary keys.
**Prevention:** Always validate dictionary keys using `str.isidentifier()` when interpolating them as column names into dynamic SQL queries, and ensure parameterized values are used.

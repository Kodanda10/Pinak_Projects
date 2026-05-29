
## 2024-05-29 - Prevent SQL Injection and Mass Assignment in Dynamic Queries
**Vulnerability:** The `update_memory` method dynamically constructed SQL queries `f"UPDATE {table} SET {set_clause}"` where `set_clause` was built from user-provided dictionary keys. Because `updates.keys()` was used directly, an attacker could potentially pass a key like `column = 1, other_column = ?; DROP TABLE memories_semantic; --` bypassing parameterization.
**Learning:** Python dictionary keys are not necessarily safe to use directly in string formatting for SQL queries when dynamic query construction is needed. Parameterization protects values, but the schema identifiers (columns/tables) still require validation if user-controlled.
**Prevention:** Always validate that any user-controlled string used as a column or table name is a valid identifier using `isinstance(key, str)` and `key.isidentifier()` before inserting it into an SQL template string.

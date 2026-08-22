## 2024-08-22 - SQL Injection in Dynamic UPDATE Queries
**Vulnerability:** The `update_memory` function constructed a SQL `SET` clause using unescaped dictionary keys from the `updates` payload (`f"{k} = ?"`). An attacker could inject arbitrary SQL by providing a crafted key in the JSON payload.
**Learning:** Even when values are parameterized, dynamically constructing SQL statements (like column names in `SET` or `WHERE` clauses) using user-controlled dictionary keys creates a SQL injection vulnerability.
**Prevention:** Always explicitly quote and escape dynamic identifiers when constructing SQL. For SQLite, column names should be wrapped in double quotes, and internal double quotes must be escaped (e.g., `'"{}".format(k.replace('"', '""'))`).

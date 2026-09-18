## 2024-05-24 - SQL Injection in Dynamic Column Names
**Vulnerability:** The `update_memory` function allowed unsanitized dictionary keys to be directly interpolated into the SQL `SET` clause, creating a SQL injection vulnerability.
**Learning:** Dynamically building SQL statements from user-provided dictionary keys without explicit validation allows attackers to inject arbitrary SQL logic, even when using parameterized queries for values.
**Prevention:** Always strictly validate dynamically generated column names (e.g., using `re.match(r"^[a-zA-Z0-9_]+$", key)`) before including them in a SQL query via string interpolation.

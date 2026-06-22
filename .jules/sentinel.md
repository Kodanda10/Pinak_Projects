## 2026-06-22 - Critical SQL Injection via Dynamic Dictionary Keys
**Vulnerability:** A critical SQL injection vulnerability was found in the DatabaseManager where dictionary keys from the 'updates' payload were interpolated directly into the SET clause, allowing attackers to overwrite the WHERE constraints.
**Learning:** Application-level key filtering is insufficient if the database layer itself dynamically constructs SQL queries with arbitrary dictionary keys.
**Prevention:** Always validate all dynamically interpolated column names and dictionary keys using `str.isidentifier()` at the exact location where the database query is executed.

## 2024-03-24 - SQL Injection in Memory Updates
**Vulnerability:** SQL Injection in `update_memory` function via unvalidated dictionary keys in the `updates` payload being directly interpolated into the `SET` clause string.
**Learning:** Even when parameterized queries are used for values, interpolating user-controlled keys directly into the SQL statement string allows attackers to inject malicious SQL syntax or bypass `WHERE` clauses by commenting them out.
**Prevention:** Ensure any dynamically generated column names are strictly validated against an alphanumeric regex (e.g. `^[a-zA-Z0-9_]+$`) or checked against a hardcoded schema whitelist before being interpolated into the query string.

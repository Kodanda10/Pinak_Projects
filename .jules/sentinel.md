## 2025-02-14 - Fix SQL Injection in SQLite update query vector

**Vulnerability:** User input from payload keys was directly interpolated via f-strings into an SQLite update query in the `update_memory` method (flagged as B608 by Bandit).
**Learning:** Keys of dynamic objects provided by users were used to construct SQL `SET` clauses directly. This creates a risk where crafted payload keys can execute arbitrary SQL injection if parameterization is bypassed.
**Prevention:** Keys from payloads that are used to build SQL statements dynamically must be sanitized. They should be explicitly double-quoted and have internal double-quotes properly escaped to securely handle untrusted names in SQLite when constructing dynamic update statements.

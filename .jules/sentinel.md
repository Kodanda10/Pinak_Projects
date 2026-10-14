## 2026-10-14 - Fix SQL injection in dynamic update queries
**Vulnerability:** SQL injection vulnerability in `DatabaseManager.update_memory` where user-supplied dictionary keys were directly interpolated into the SQL `SET` clause without validation, and sensitive fields could be modified.
**Learning:** Dynamic column generation from user input (e.g. from an API request) must be validated to ensure keys are valid identifiers, otherwise attackers can inject arbitrary SQL. Additionally, IDOR prevention requires explicitly stripping protected fields before update.
**Prevention:** Always validate dynamic column names using `.isidentifier()` and filter out protected multi-tenant/id fields before constructing query clauses.

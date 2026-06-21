## 2026-06-21 - SQL Injection in update_memory
**Vulnerability:** SQL injection vector found via Bandit scan in update_memory function where dictionary keys are used to build SET clause dynamically.
**Learning:** The database layer dynamically constructs queries from unsanitized dictionary keys, leading to potential SQL injection. Application-level key filters are insufficient.
**Prevention:** Validate all dictionary keys using str.isidentifier() before using them to construct SQL queries.

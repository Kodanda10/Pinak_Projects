## 2024-03-12 - Fix SQL Injection in update_memory
**Vulnerability:** SQL injection and mass assignment via unsanitized dictionary keys in dynamic SQL UPDATE queries (`update_memory`).
**Learning:** Dynamic column names generated from unsanitized input (`updates.keys()`) can execute arbitrary SQL injection payloads since parameterized queries (`?`) only protect values, not structural identifiers.
**Prevention:** Strictly validate dynamic column names using `.isidentifier()` and an explicit whitelist of allowed updatable fields, ignoring protected fields like `id` and `tenant`.

## 2025-02-27 - SQL Injection in update_memory
**Vulnerability:** SQL injection via unsanitized dictionary keys in update_memory.
**Learning:** Dictionary keys were used to construct SQL SET clauses directly.
**Prevention:** Validate dictionary keys to ensure they are valid Python identifiers.

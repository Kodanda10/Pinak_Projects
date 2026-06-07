## 2026-06-07 - SQL Injection in dynamic SET clause
**Vulnerability:** SQL Injection via unsanitized dictionary keys used to construct the SET clause in update_memory.
**Learning:** Dynamic query construction using user-supplied dictionaries without key validation allows arbitrary SQL injection and mass assignment.
**Prevention:** Enforce that dictionary keys used in queries are valid strings and Python identifiers using isidentifier().

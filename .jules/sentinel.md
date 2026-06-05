## 2026-06-05 - [SQL Injection via Dynamic Updates]
**Vulnerability:** SQL Injection and mass assignment via unsanitized dictionary keys in the dynamic UPDATE query generation inside update_memory.
**Learning:** Constructing dynamic SQL queries from unsanitized user dictionary keys exposes the database to injection attacks, even if values are parameterized.
**Prevention:** Enforce that all dictionary keys used in dynamic query generation are valid Python strings and valid identifiers (e.g., using str.isidentifier()) before constructing queries.

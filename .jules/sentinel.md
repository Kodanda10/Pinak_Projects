## 2026-06-08 - [Strict Validation for Dynamic SQL Keys]
**Vulnerability:** SQL injection and mass assignment via unsanitized dictionary keys in dynamic UPDATE queries.
**Learning:** Dynamic SQL string concatenation (e.g. `set_clause`) using dictionary keys can be exploited if the keys aren't strictly checked, bypassing parameterized values.
**Prevention:** Always enforce that dictionary keys are valid strings and Python identifiers using `isidentifier()` before constructing dynamic SQL queries.

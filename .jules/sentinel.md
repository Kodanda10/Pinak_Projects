## 2024-07-19 - SQL Injection via Unvalidated Keys in update_memory
**Vulnerability:** Dynamic SQL generation in `DatabaseManager.update_memory` iterates over `updates` dictionary keys to construct a `SET` clause without validating the keys, allowing SQL injection via crafted payload keys.
**Learning:** Dynamic query construction using external data structures must validate all identifiers, even if parameterized values are used, as column names cannot be parameterized. Bandit does not always trace complex data flows or dynamic validations to catch these issues.
**Prevention:** Always validate dynamically constructed SQL identifiers using `str.isidentifier()` and strictly allowlist expected columns where possible.

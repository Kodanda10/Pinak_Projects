## 2024-05-18 - Fix SQL Injection in DatabaseManager.update_memory
**Vulnerability:** SQL Injection and mass assignment via dynamic SQL query construction.
**Learning:** It existed because `update_memory` directly looped over the dictionary `updates` to build a dynamic SQL query without checking if the keys were safe column names, making it vulnerable to injection attacks if keys are malicious.
**Prevention:** Ensure dictionary keys used for building dynamic queries are validated as valid Python identifiers (e.g., `not isinstance(key, str) or not key.isidentifier()`) before constructing the query.

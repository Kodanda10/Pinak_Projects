## 2024-03-12 - Fix Mass Assignment and SQL Injection in `update_memory`
**Vulnerability:** The `update_memory` function across both the `MemoryService` and `DatabaseManager` suffered from two vulnerabilities:
1. Mass Assignment: The `MemoryService` allowed any field not explicitly in a tiny blocklist to be updated.
2. SQL Injection / Logic Error: The `DatabaseManager` dynamically constructed the `UPDATE` SQL statement by iterating over arbitrary keys provided in the `updates` dictionary, allowing string values to be placed directly into the query clause structure.
**Learning:** Security blocklists are fragile because as schemas grow, any un-blocklisted field automatically becomes mutable. Directly unpacking dictionary keys into a query string is dangerous, even with prepared parameters for values.
**Prevention:**
1. Use strict allowlists (whitelists) for field updates based on the exact layer schema.
2. Validate dictionary keys used for dynamic query building by enforcing `isinstance(key, str) and key.isidentifier()`.

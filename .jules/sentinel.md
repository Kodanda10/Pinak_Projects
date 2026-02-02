# Sentinel's Journal

## 2025-05-15 - Mass Assignment via SQL Injection in DatabaseManager
**Vulnerability:** Found a mass assignment vulnerability in `DatabaseManager.update_memory` where the `updates` dictionary keys were directly interpolated into the SQL `SET` clause without validation. This allowed bypassing filters (like `MemoryService`'s check for `tenant`) by injecting combined keys (e.g., `"content = 'x', tenant"`), leading to Tenant Hijacking.
**Learning:** Checking for forbidden keys (`if k not in forbidden`) is insufficient when keys are used in string interpolation for SQL. An attacker can use a key that *contains* valid SQL to manipulate the query structure while evading the exact-match filter.
**Prevention:** Always validate dynamic keys against a strict allowlist (whitelist) of valid column names before using them in SQL queries. Never rely on blacklists for security-critical input validation.

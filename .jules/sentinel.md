## 2026-02-16 - Dynamic SQL Key Injection in Memory Update
**Vulnerability:** SQL Injection in `DatabaseManager.update_memory` where dictionary keys were directly interpolated into the `SET` clause of an `UPDATE` statement.
**Learning:** Checking against a blacklist of forbidden keys (`MemoryService.forbidden_keys`) is insufficient because attackers can inject keys that pass the blacklist but contain SQL fragments (e.g., `"embedding_id = 999, tags"`).
**Prevention:** Always validate dynamic dictionary keys against a strict whitelist of existing table columns (using `PRAGMA table_info`) before using them in SQL construction.

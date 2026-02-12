## 2026-02-12 - DatabaseManager Dynamic Query Vulnerability
**Vulnerability:** The `DatabaseManager` class constructs SQL `UPDATE` statements dynamically using keys from the input dictionary without validation. This allows mass assignment vulnerabilities if the service layer fails to strictly filter inputs.
**Learning:** In `MemoryService.update_memory`, a blacklist approach was used (`forbidden_keys`), which failed to protect sensitive fields like `client_id` or `agent_id` that were not in the blacklist.
**Prevention:** Always use a strict **whitelist** approach for input validation, especially when the underlying data layer is permissive. The `DatabaseManager` should ideally also validate keys against a schema, but service-layer validation is the primary defense.

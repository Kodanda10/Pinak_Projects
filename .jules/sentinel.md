## 2024-06-25 - SQL Injection in Memory Service `update_memory`
**Vulnerability:** The `update_memory` method in `Pinak_Services/memory_service/app/core/database.py` dynamically builds a `SET` clause from `updates.keys()` without strictly validating if those keys are valid identifiers, opening up potential SQL injection vectors.
**Learning:** Even though `memory_service.py` filters some `forbidden_keys`, it doesn't validate if a key is a proper identifier. This allows malicious keys like `bad_key = ? --` to bypass subsequent constraints or inject malicious SQL. The database layer must independently guarantee identifier safety.
**Prevention:** Validate all dictionary keys that dynamically construct SQL queries using `str.isidentifier()` at the exact point of SQL string construction.

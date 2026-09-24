## 2024-09-24 - Dynamic SQL Injection in Memory Updates
**Vulnerability:** A SQL injection vulnerability exists in `update_memory` within `Pinak_Services/memory_service/app/core/database.py` due to dynamically concatenating unvalidated dictionary keys into the `UPDATE` query's `SET` clause.
**Learning:** Even if updates are partially filtered elsewhere, constructing raw SQL queries from dictionary keys without asserting they are safe column names allows malicious payloads to bypass parameterization and alter the query structure.
**Prevention:** Always validate dynamically generated column names using Python's `.isidentifier()` method or against an explicit allowlist before using them in raw SQL strings.

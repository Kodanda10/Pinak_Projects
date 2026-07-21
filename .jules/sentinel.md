## 2024-07-21 - SQL Injection in update_memory
**Vulnerability:** Dynamic SQL query construction in `DatabaseManager.update_memory` used unsanitized dictionary keys from `updates`.
**Learning:** Bandit SAST scans might not catch dynamic SQL construction vulnerabilities correctly when f-strings are used for keys. Unsanitized keys can lead to SQL injection.
**Prevention:** Always validate dynamically provided column names using `str.isidentifier()` before using them in SQL queries.

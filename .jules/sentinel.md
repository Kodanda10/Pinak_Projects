## 2024-05-24 - Fix SQL Injection in Memory Updates
**Vulnerability:** Dynamic SQL constructed directly from user-provided dictionary keys in update_memory.
**Learning:** Parameterized queries only protect values, not table or column names.
**Prevention:** Always validate keys dynamically used in SQL as strings and valid Python identifiers.

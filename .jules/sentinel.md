## 2023-10-25 - Fix SQL Injection in DatabaseManager.update_memory
**Vulnerability:** In `DatabaseManager.update_memory`, dictionary keys used to dynamically construct SQL statements (`SET` clauses) were unquoted and vulnerable to SQL injection from untrusted user payloads.
**Learning:** When dynamically constructing SQL `SET` clauses from dictionary keys, it's critical to properly quote the keys.
**Prevention:** Always quote dictionary keys used to dynamically construct SQL statements with double quotes (e.g., `"\"{k}\" = ?"`).

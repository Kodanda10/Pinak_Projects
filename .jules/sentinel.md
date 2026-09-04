## 2024-09-04 - SQL Injection Vulnerability in DatabaseManager.update_memory
**Vulnerability:** Untrusted dictionary keys from user payloads are directly interpolated into the SQL `UPDATE` statement's `SET` clause without proper escaping or quotation (`f"{k} = ?"`), which enables SQL injection attacks.
**Learning:** Even when values are correctly parameterized, dynamic generation of SQL clauses based on untrusted keys (like dictionary keys representing column names) must explicitly escape and quote those keys.
**Prevention:** Always strictly validate, sanitize, or quote dynamic SQL identifiers (like column or table names). For dictionary keys used as column names, double-quote the key and escape internal double quotes (e.g., `'"{}" = ?'.format(k.replace('"', '""'))`).

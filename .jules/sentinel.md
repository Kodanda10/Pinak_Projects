## YYYY-MM-DD - Mitigated SQL Injection & Mass Assignment in Updates
**Vulnerability:** `DatabaseManager.update_memory` took an arbitrary `updates` dictionary and interpolated the keys directly into the SQL query: `f"UPDATE {table} SET {k} = ?"`. This exposed the system to SQL injection via column name manipulation and mass assignment vulnerabilities.
**Learning:** SQLite prevents multiple statements but allowing arbitrary keys in the SET clause can lead to column name manipulation.
**Prevention:** Enforce string type and `.isidentifier()` validation on all keys entering the `DatabaseManager` update query construction.

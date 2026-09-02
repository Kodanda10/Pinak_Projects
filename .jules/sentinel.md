## 2026-09-02 - SQL Injection in Memory Update
**Vulnerability:** The memory update method in `DatabaseManager` constructed the SQL `SET` clause by dynamically iterating over unescaped dictionary keys from the user payload (`updates`). A malicious key could bypass the parameterized variables and inject arbitrary SQL logic.
**Learning:** Even when utilizing parameterized SQL for values, if table structure components like column names or `SET` keys are dynamic and driven by user input, they must be explicitly sanitized, double-quoted, and escaped to prevent injection.
**Prevention:** Always strictly double-quote and escape dynamic keys (e.g., `'"{}" = ?'.format(k.replace('"', '""'))`) within SQL queries if they originate from an external, untrusted source payload.

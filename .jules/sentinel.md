## 2024-07-17 - Dynamic SQL Injection in update_memory
**Vulnerability:** SQL injection vulnerability in `DatabaseManager.update_memory` due to unsanitized dictionary keys used to construct dynamic `UPDATE` queries.
**Learning:** Even though `conn.execute()` blocks multiple statement execution, single-statement SQL injections can still bypass trailing `WHERE` constraints by using comment markers (`--`) or balancing unused bindings.
**Prevention:** Always validate dynamically injected dictionary keys in SQL queries using `str(key).isidentifier()` to ensure they are valid SQL identifiers, preventing malicious payloads.

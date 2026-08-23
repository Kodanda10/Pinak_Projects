## 2024-08-23 - Fix SQL Injection in DatabaseManager.update_memory
**Vulnerability:** SQL injection vulnerability in `DatabaseManager.update_memory` due to unsanitized user payload keys being interpolated into the SET clause.
**Learning:** Even when values are parameterized, dictionary keys used to dynamically construct SQL queries must be explicitly sanitized, especially when derived from untrusted user input payloads.
**Prevention:** Explicitly double-quote and escape internal double-quotes for dynamic SQL identifiers (e.g. `'"{}" = ?'.format(k.replace('"', '""'))`) and combine with parameterized queries for values.

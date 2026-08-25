## 2024-08-25 - SQL injection in dynamic SET clauses
**Vulnerability:** SQL injection vulnerability in `DatabaseManager.update_memory` caused by directly formatting untrusted JSON dictionary keys into a dynamic SQL `SET` clause.
**Learning:** The codebase builds SQL queries dynamically from untrusted JSON payloads to accommodate flexible schemas, inadvertently bypassing parameterization protections which only applied to the dictionary values.
**Prevention:** Always explicitly double-quote and escape internal quotes for dynamically inserted column names (e.g., `'"{}" = ?'.format(k.replace('"', '""'))`) when parameterization cannot be used for identifiers.

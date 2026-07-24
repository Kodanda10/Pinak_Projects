
## 2026-07-24 - Prevent SQL injection in dynamic queries
**Vulnerability:** SQL injection vulnerability in `DatabaseManager.update_memory` due to unsanitized dictionary keys used to construct `UPDATE` queries.
**Learning:** Python dictionaries used to build dynamic SQL queries can lead to SQL injection if keys are not validated, even if values are parameterized.
**Prevention:** Always validate dynamically constructed SQL identifiers (like column names) against `str.isidentifier()` before incorporating them into the query.

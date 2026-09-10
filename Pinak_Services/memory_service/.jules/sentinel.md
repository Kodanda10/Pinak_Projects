## 2024-10-24 - Fix SQL Injection in DatabaseManager.update_memory

**Vulnerability:** A critical SQL injection was identified in `DatabaseManager.update_memory` where user-controlled dictionary keys were unescaped and directly interpolated into the `SET` clause of a dynamically built `UPDATE` statement.
**Learning:** Generating dynamic SQL queries from dictionaries can allow SQL injection if the keys are not properly quoted and escaped.
**Prevention:** Ensure that all user-supplied input used as SQL identifiers (such as column names) is strictly double-quoted, with internal quotes properly escaped. Using parameter binding is not possible for column identifiers.

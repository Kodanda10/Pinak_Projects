## 2024-05-18 - SQL Injection in Memory Update Keys
**Vulnerability:** Untrusted dictionary keys used directly in dynamically constructed SQL `UPDATE` statement in `DatabaseManager.update_memory`.
**Learning:** Even when parameterized bindings are used for values, dynamically inserting dictionary keys into `SET` clauses without escaping can lead to SQL injection vulnerabilities, allowing attackers to manipulate queries.
**Prevention:** Always double-quote column identifiers dynamically constructed from user input and properly escape internal double quotes (e.g. `k.replace('"', '""')`). Use `# nosec B608` to suppress false positive Bandit warnings when necessary.

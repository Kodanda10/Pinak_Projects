## 2026-08-20 - Prevent SQL Injection via dynamically built queries
**Vulnerability:** Constructing SQL `SET` clauses or table names dynamically using string formatting with user inputs in SQLite creates an SQL injection vulnerability (Bandit B608).
**Learning:** Always double quote keys in dynamically generated SQL `SET` clauses. Additionally, when dynamic queries use validated whitelisted tables, Bandit B608 throws false positives.
**Prevention:** Use strictly typed whitelists for table/column names. Explicitly quote and escape keys for dynamic statements `"{k}" = ?`. Use `# nosec B608` to suppress static analysis warnings on validated whitelists.

## 2023-10-25 - Prevent SQL Injection via Dictionary Key Mass Assignment
**Vulnerability:** The `update_memory` function iterated over arbitrary dict keys provided in the `updates` JSON payload, concatenating them directly into an SQL `SET` clause string: `set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])`. This permitted SQL injection if malicious keys were passed (e.g., `content = 'hacked' --`).
**Learning:** Even when using parameterized values (`?`), the keys composing the schema structure are vulnerable to injection if derived from untrusted client input.
**Prevention:** Always validate dynamically generated column names in SQL queries using strict `.isidentifier()` checks and whitelists to ensure only safe, permitted schema columns are modified.

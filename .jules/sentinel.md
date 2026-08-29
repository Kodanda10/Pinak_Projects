## 2024-05-24 - Dynamic Column Name SQL Injection
**Vulnerability:** Dictionary keys from untrusted user payloads were directly interpolated into SQL string `SET` clauses during memory updates, enabling SQL injection despite the values themselves being parameterized.
**Learning:** Even when using parameterized queries for values, dynamic column names derived from dictionary keys must be explicitly secured if the dictionary originates from an untrusted source.
**Prevention:** Always explicitly double-quote and escape dictionary keys when used as dynamic SQL column names (e.g., `'"{}" = ?'.format(k.replace('"', '""'))`).

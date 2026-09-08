
## 2024-09-08 - SQL Injection in DatabaseManager.update_memory
**Vulnerability:** The memory update method dynamically constructs SET clauses via vulnerable string concatenation for dictionary keys (`set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])`), making it susceptible to SQL injection from untrusted key names.
**Learning:** Even when values are parameterized in SQLite via `?`, dynamically injected column or JSON key names must still be properly escaped to avoid tampering in dynamic queries.
**Prevention:** For dynamically generated queries, explicitly double-quote untrusted keys and escape internal double-quotes (e.g., `"{k.replace('"', '""')}"`) to ensure they are strictly treated as identifiers, not syntax.

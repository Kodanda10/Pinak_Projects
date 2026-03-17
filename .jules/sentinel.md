## 2024-05-24 - Fix SQL injection warnings in database.py
**Vulnerability:** Bandit detected B608 (hardcoded_sql_expressions) in python strings being formatted directly into SQL queries (e.g., `f"SELECT * FROM {table} ..."`).
**Learning:** These tables names were coming from hardcoded maps (e.g. `table_map`), so they are intrinsically safe from direct SQL injection via user input.
**Prevention:** We can safely append `# nosec B608` to explicitly suppress the false positive warnings and document that these strings are safe.
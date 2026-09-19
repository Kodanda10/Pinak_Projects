## 2026-09-19 - SQL Injection in update_memory
**Vulnerability:** The `update_memory` function in `Pinak_Services/memory_service/app/core/database.py` constructs a SQL query using string concatenation for column names directly from an unvalidated `updates` dictionary: `set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])`.
**Learning:** This exposes the application to SQL injection since dictionary keys aren't parameterized like values are. This existed because dynamic update clauses require manual column validation.
**Prevention:** Always strictly validate dynamic column names against an allowlist or alphanumeric regex (e.g., `re.match(r"^[a-zA-Z0-9_]+$", key)`) before string interpolation into SQL queries.

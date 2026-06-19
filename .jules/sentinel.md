## 2024-06-19 - SQL Injection in DatabaseManager.update_memory
**Vulnerability:** SQL injection vulnerability in `DatabaseManager.update_memory` caused by dynamically interpolating unsanitized dictionary keys into the SET clause. This allows attackers to bypass isolation (e.g., tenant, project_id) by crafting malicious keys.
**Learning:** Application-level key filters (if any) do not prevent SQL injection or mass assignment if the underlying database layer (`app/core/database.py`) dynamically constructs queries from unsanitized dictionary keys.
**Prevention:** Always validate dictionary keys using `str.isidentifier()` at the database query execution layer to ensure they are valid column names.

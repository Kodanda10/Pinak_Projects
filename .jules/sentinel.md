## 2024-06-29 - SQL Injection in DatabaseManager.update_memory
**Vulnerability:** SQL Injection via unsanitized dictionary keys in update_memory SET clause.
**Learning:** Application-level key filters in the service layer do not prevent injection if the database layer dynamically constructs queries from unsanitized keys.
**Prevention:** Always validate dynamically interpolated column names using str.isidentifier() at the query execution layer.

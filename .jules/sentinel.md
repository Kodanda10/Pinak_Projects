## 2024-05-18 - SQL Injection in Dynamic UPDATE Statements
**Vulnerability:** SQL injection vulnerability via dynamically constructed UPDATE queries in `DatabaseManager.update_memory`, because dictionary keys bypassed SQL parameterization.
**Learning:** Application-level key filters don't prevent SQL injection if the underlying database layer uses unsanitized keys in queries.
**Prevention:** Always validate dynamically incorporated identifier names using `str.isidentifier()`.

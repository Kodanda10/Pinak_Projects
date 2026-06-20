## 2026-06-20 - Fix SQL Injection in Database Layer
**Vulnerability:** SQL injection vulnerability in `DatabaseManager.update_memory` due to unsanitized dictionary keys dynamically interpolated into the SET clause.
**Learning:** Even if the application layer (like `memory_service.py`) filters some keys, the underlying database layer must strictly validate all input used to construct SQL statements dynamically to prevent SQL injection or mass assignment.
**Prevention:** Always validate dynamically interpolated dictionary keys using strict rules like `str.isidentifier()` or against an allowlist in the database query execution layer.

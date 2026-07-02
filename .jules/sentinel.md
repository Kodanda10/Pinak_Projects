## 2026-07-02 - Prevent SQL Injection in Database Layer
**Vulnerability:** Dynamic SQL SET clause construction in `update_memory` used unsanitized dictionary keys, allowing SQL injection even if application-layer filters exist.
**Learning:** Application-level key filters (`forbidden_keys`) are insufficient defense in depth. The database layer must independently validate dynamically interpolated schema components.
**Prevention:** Always validate dictionary keys using `str.isidentifier()` at the database query execution layer when constructing queries dynamically.

## 2024-05-15 - Fixed SQL Injection and IDOR in update_memory
**Vulnerability:** SQL Injection in dynamic query generation where user-controlled dictionary keys were injected directly into the SET clause without validation. Also IDOR risk where protected fields like id and tenant could be overwritten.
**Learning:** Dynamic column construction in Python using f-strings is a common vector for SQLi when inputs aren't validated. Filtering keys via .isidentifier() provides robust defense for Python-based dynamic query builder structures.
**Prevention:** Always validate dynamically constructed keys via strict allowlisting or pattern-matching (e.g. .isidentifier()) and explicitly exclude protected metadata fields from updates.

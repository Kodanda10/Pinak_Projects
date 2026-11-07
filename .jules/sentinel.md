## 2023-11-07 - [Dynamic Column Generation Bypass]
**Vulnerability:** Mass assignment and malicious column injection
**Learning:** A blocklist check at the application layer can be bypassed if the underlying database tier iterates directly over dictionary keys without strictly validating against allowed identifiers. Keys like `tenant = 'new', content` are not in the blocklist but alter unintended columns.
**Prevention:** Always use a strict allowlist and `.isidentifier()` when dynamically generating column updates in the database tier.

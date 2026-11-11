## 2024-05-15 - Mass Assignment in dynamic SQL updates
**Vulnerability:** Dynamic column generation in update queries allows mass assignment and restricted column updates.
**Learning:** Python dictionaries passed from untrusted input directly to SQL UPDATE structures bypass application-level protections for protected fields like 'tenant' and 'id'.
**Prevention:** Use .isidentifier() and an explicit strict whitelist for column generation in dynamic SQL.

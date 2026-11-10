## 2024-11-10 - Fix dynamic column generation in update_memory
**Vulnerability:** Dynamic column generation in update_memory allowed mass assignment by trusting dictionary keys.
**Learning:** Even when using parameterized queries for values, using unsanitized user input for column names in statements introduces critical risks.
**Prevention:** Always validate dynamically generated column names against a strict whitelist of allowed identifiers and ensure they comply with str.isidentifier().

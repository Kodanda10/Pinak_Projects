## 2024-05-18 - Prevent SQL Injection via Dynamic Column Updates
**Vulnerability:** Dynamic SQL update statements were constructed using unsanitized dictionary keys, allowing potential SQL injection and mass assignment.
**Learning:** Even internal dictionaries mapped from external input must be strictly validated before being interpolated into SQL statements, specifically when constructing the SET clause.
**Prevention:** Always validate dynamically generated column names using strict whitelists and `.isidentifier()` checks to ensure they are valid SQL identifiers and do not contain malicious payloads.

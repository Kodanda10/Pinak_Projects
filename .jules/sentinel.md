## 2024-05-10 - SQL Injection via Dictionary Keys in Dynamic Update Queries
**Vulnerability:** Possible SQL injection vector in `update_memory` due to dynamic SQL generation building a `SET` clause using unsanitized dictionary keys from the `updates` parameter.
**Learning:** Constructing SQL queries dynamically using `set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])` is risky when keys are unverified, as malicious keys can bypass prepared statement parameterization.
**Prevention:** Always validate that dictionary keys are strings and valid Python identifiers using `isinstance(key, str) and key.isidentifier()` before utilizing them to build dynamic SQL strings.

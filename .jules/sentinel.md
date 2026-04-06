## 2025-04-06 - Remove Hardcoded JWT Default and Fix Update SQL Injection

**Vulnerability:**
The `cli/main.py` explicitly fell back to a hardcoded string (`"dev-secret-change-me"`) for the JWT secret if the `PINAK_JWT_SECRET` environment variable was not found. This allows minting and verifying tokens locally without requiring the user to actually set an environment variable, potentially leading to hardcoded secrets slipping into higher environments. Additionally, `app/core/database.py` dynamically generated a set-clause using the keys of a payload in its `update_memory` function, leading to potential SQL injection.

**Learning:**
Falling back to a "dev" secret, although well-intentioned for a smooth local developer experience, compromises the security baseline. Developers might never configure proper secrets and deploy insecure applications. Similarly, Python's parameterized `execute(query, params)` protects against values, but does not sanitize dynamically injected keys (identifiers).

**Prevention:**
Always fail securely when configuration variables are missing. Explicitly require authentication secrets to be set in the environment, rather than providing insecure "dev" fallbacks. Furthermore, always ensure any user input used as a query identifier (e.g., table or column name) is validated as a strict Python identifier (`key.isidentifier()`).
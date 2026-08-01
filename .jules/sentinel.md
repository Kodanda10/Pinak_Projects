## 2026-08-01 - Hardcoded JWT Secrets
**Vulnerability:** Hardcoded JWT secrets (e.g., "secret" and "dev-secret-change-me") were used as fallbacks across multiple CLI tools and scripts (cli/main.py, client/pinak_memory_mcp.py, shell scripts).
**Learning:** Default fallback secrets intended for development are often inadvertently carried into production or expose systems to predictable credential attacks if the environment variable is not explicitly set.
**Prevention:** Always enforce the explicit configuration of secrets via environment variables by raising exceptions or errors when they are missing. Avoid providing default values for sensitive credentials in code.

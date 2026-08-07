## 2025-02-27 - SQL Injection Vulnerabilities in Table Name Interpolation
**Vulnerability:** Found multiple instances where table names in SQLite queries were dynamically interpolated using f-strings (e.g., `f"SELECT count(*) FROM {table}"`) without proper validation or parameterization, presenting a potential SQL injection vector if `table` comes from an untrusted source, as identified by Bandit rule B608.
**Learning:** SQLite parameterization (using `?` or named parameters) only works for values, not for schema identifiers like table or column names. The application frequently loops over layer names and interpolates them as table names.
**Prevention:** Always validate dynamically constructed table and column names against a strict whitelist (e.g., predefined sets of allowed tables or columns) before using them in f-strings for SQL execution to prevent injection attacks while accommodating SQLite's limitations.

## 2025-02-27 - Hardcoded Fallback Secrets for JWT Minting
**Vulnerability:** The CLI commands `mint` and `search` in `cli/main.py`, as well as `client/pinak_memory_mcp.py` and `scripts/ingest_*.py`, default to `dev-secret-change-me` or `secret` if `PINAK_JWT_SECRET` is missing. This could lead to minting widely accepted valid tokens in production if the environment variable isn't correctly exported in that specific context (e.g., shell script or cron job).
**Learning:** Fallback JWT secrets completely negate the security provided by environment variables because the application continues to run in an insecure state instead of failing-fast and notifying the operator.
**Prevention:** Remove fallback hardcoded defaults for sensitive keys. Require them strictly via runtime checking or throwing an exception.

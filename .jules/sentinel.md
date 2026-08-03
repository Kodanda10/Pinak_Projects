## 2024-08-04 - Hardcoded JWT Secret Fallbacks
**Vulnerability:** Found hardcoded fallback secrets like `"dev-secret-change-me"` or `"secret"` used when `PINAK_JWT_SECRET` was not provided in `cli/main.py`, `scripts/ingest_history.py`, `scripts/ingest_all.py`, and `client/pinak_memory_mcp.py`.
**Learning:** Hardcoded default secrets are a significant risk. If the system goes to production without explicit secret configuration, attackers can guess the default secret and mint unauthorized JWTs.
**Prevention:** Ensure that critical secrets, like JWT secrets, are strictly required via environment variables. The system should crash or fail early if they are not explicitly provided, rather than quietly falling back to a hardcoded default.

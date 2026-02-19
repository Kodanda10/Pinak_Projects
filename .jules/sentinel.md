## 2026-02-19 - Hardcoded JWT Secrets in CLI and Scripts
**Vulnerability:** The CLI tool `pinak-memory-cli` and client scripts defaulted to `dev-secret-change-me` or `secret` if `PINAK_JWT_SECRET` was not set. This allowed users to inadvertently deploy or use the service with a weak, known secret.
**Learning:** Convenience defaults in CLI tools can undermine server security if they mismatch or are weak. "Safe by default" means failing if a secret is missing, or generating a strong one if appropriate (e.g. for a self-contained server).
**Prevention:** Avoid default values for sensitive configuration. Use `required=True` or raise explicit errors. For standalone servers, auto-generate strong secrets on startup if missing.

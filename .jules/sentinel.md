## 2025-02-12 - Hardcoded JWT Secret Default
**Vulnerability:** A hardcoded default secret (`"secret"` or `"dev-secret-change-me"`) was used in several places for JWT tokens across the CLI and client tools.
**Learning:** Fallback secrets allow easy generation of valid JWT tokens if the environment variable is not explicitly set, completely bypassing authentication security in production if deployed with defaults.
**Prevention:** Always require sensitive environment variables like `PINAK_JWT_SECRET` to be set explicitly and avoid using default fallback secrets in production code or scripts.

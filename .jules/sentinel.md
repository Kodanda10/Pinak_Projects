## 2025-03-05 - Hardcoded JWT Secrets in CLI and Scripts
**Vulnerability:** Hardcoded fallback values for PINAK_JWT_SECRET ("dev-secret-change-me" and "secret") were used in CLI tools and scripts.
**Learning:** Providing default fallback secrets compromises security by allowing authentication with known credentials if the environment variable is inadvertently omitted.
**Prevention:** Ensure all JWT and cryptographic secrets are strictly required via environment variables with no fallback values, raising an explicit error if missing.

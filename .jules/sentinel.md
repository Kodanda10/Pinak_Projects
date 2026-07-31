## 2024-05-24 - Hardcoded Fallback Secrets for JWT
**Vulnerability:** Hardcoded fallback secrets (e.g., `dev-secret-change-me`) were used for JWT generation in CLI tools.
**Learning:** Missing environment variables must raise an exception rather than falling back to an insecure default, which can lead to unauthorized access.
**Prevention:** Enforce strict presence checks for critical secrets (like `PINAK_JWT_SECRET`) and fail fast if missing.

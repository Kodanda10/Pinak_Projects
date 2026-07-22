## 2024-07-22 - Hardcoded JWT Secret in CLI
**Vulnerability:** Hardcoded fallback JWT secret "dev-secret-change-me" used in CLI token generation and requests.
**Learning:** Default development secrets were left in production-facing CLI code, allowing potential token forgery if used against an exposed instance with the same default.
**Prevention:** Always require secrets to be explicitly provided via environment variables or arguments; never hardcode fallback cryptographic keys.

## 2026-07-29 - Hardcoded JWT Secrets

**Vulnerability:** The CLI tooling falls back to a hardcoded default secret (`dev-secret-change-me`) when `PINAK_JWT_SECRET` is omitted.
**Learning:** Hardcoded default secrets undermine the security of JWT authentication because they allow attackers to mint valid tokens if the environment variable is accidentally omitted, breaking the strict multi-tenancy enforced by the JWT.
**Prevention:** Always strictly require the `PINAK_JWT_SECRET` environment variable for all JWT generation and authentication, and explicitly raise an exception or abort if it is not set. Do not use hardcoded defaults in code.

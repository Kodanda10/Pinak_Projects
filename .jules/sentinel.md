## 2024-05-31 - Removed Hardcoded JWT Fallback Secrets
**Vulnerability:** Found multiple CLI scripts and internal tools using hardcoded fallback JWT secrets (e.g., `dev-secret-change-me` or `secret`) when the environment variable was missing.
**Learning:** In development environments, it is common to fallback to a known string to make things run smoothly out-of-the-box, but these hardcoded values often leak into production setups or are mistakenly used by end-users.
**Prevention:** Strict enforcement of the `PINAK_JWT_SECRET` environment variable must be enforced across all components, even in CLI and auxiliary scripts, without relying on default string fallbacks. Missing environment variables should loudly fail instead of degrading into an insecure state.

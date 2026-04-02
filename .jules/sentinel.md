## 2025-04-02 - Remove hardcoded secret default
**Vulnerability:** The CLI default fallback `dev-secret-change-me` was being used when `PINAK_JWT_SECRET` was not set.
**Learning:** Hardcoded default secrets can be accidentally deployed or exploited if users forget to set the environment variable.
**Prevention:** Applications must strictly require security environment variables to be set and fail securely if they are not, instead of providing insecure fallbacks.

## 2025-02-20 - Remove hardcoded JWT secrets
**Vulnerability:** Hardcoded JWT secret fallbacks (e.g. 'dev-secret-change-me' and 'secret') were present in CLI tools and ingestion scripts.
**Learning:** Hardcoding fallback secrets compromises the security of the application and tokens if the environment variable is not explicitly provided.
**Prevention:** Always require strict enforcement of secrets via environment variables and raise an exception or exit if they are missing.

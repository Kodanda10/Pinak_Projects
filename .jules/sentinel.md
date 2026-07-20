## YYYY-MM-DD - Hardcoded JWT Secret in CLI
**Vulnerability:** Hardcoded fallback JWT secret "dev-secret-change-me" used in `Pinak_Services/memory_service/cli/main.py`.
**Learning:** Hardcoding secrets as fallbacks in CLI tools introduces severe risks. Any token minted using this fallback could be used maliciously against instances sharing the same fallback.
**Prevention:** Avoid fallback defaults for cryptographic secrets. Enforce required environment variables or flags.

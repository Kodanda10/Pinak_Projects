## 2024-07-27 - [Hardcoded JWT Secrets]
**Vulnerability:** Hardcoded fallback secrets ("dev-secret-change-me") were used in CLI tools for JWT minting and searching.
**Learning:** Fallback secrets in code can inadvertently allow unauthorized access or token generation if environment configurations are missed, especially in sensitive tools.
**Prevention:** Always require secrets to be injected via environment variables or explicit configuration, and fail securely (abort) if they are missing.

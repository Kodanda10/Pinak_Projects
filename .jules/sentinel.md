## 2025-01-26 - [Hardcoded JWT Secrets]
**Vulnerability:** Multiple CLI scripts (`cli/main.py`), tools (`client/pinak_memory_mcp.py`), and ingestion scripts used hardcoded fallback values for `PINAK_JWT_SECRET` (e.g., `"secret"` or `"dev-secret-change-me"`).
**Learning:** Developer convenience features in scripts can easily leak into production if environment variables are not strictly enforced, risking complete auth bypass.
**Prevention:** Never use hardcoded fallback values for cryptographic secrets in `os.getenv` or `os.environ.get`. Always require the variable and raise an exception if missing.

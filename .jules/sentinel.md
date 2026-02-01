## 2024-05-22 - Public Schema Endpoint Information Disclosure
**Vulnerability:** The `/api/v1/memory/schema` endpoints were publicly accessible and returned the absolute server path of the schema directory (`schema_dir`, `fallback_dir`).
**Learning:** Metadata/schema inspection endpoints are often overlooked during security passes but can leak internal environment structure (paths) which aids attackers in traversing the filesystem.
**Prevention:** Apply `require_auth_context` to all endpoints, even those providing "harmless" metadata. Strip internal server paths from API responses.

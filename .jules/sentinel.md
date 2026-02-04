## 2024-05-21 - JWT Claim Spoofing via Headers
**Vulnerability:** The `require_auth_context` dependency prioritized HTTP headers (`X-Pinak-Client-Id`) over signed JWT claims (`client_id`). This allowed authenticated users to impersonate any client by injecting a header.
**Learning:** The logic used `header or payload` pattern, assuming headers were only for "missing" data, but failed to realize headers are user-controlled and can override existing trusted data.
**Prevention:** Always strictly prioritize trusted sources (JWT, Server Session) over untrusted sources (Headers, Query Params). Use `trusted_value or untrusted_fallback`, never the reverse.

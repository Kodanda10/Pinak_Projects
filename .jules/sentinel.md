# Sentinel Journal

## 2025-10-26 - Client-Side Trust in Auth Context
**Vulnerability:** The application blindly trusted the `X-Pinak-Client-Id` header to determine the `client_id`, even when a valid JWT was provided. This allowed potential client spoofing where a malicious user could supply a valid token but impersonate a different client ID by setting the header.
**Learning:** Never trust client-provided headers for identity or authorization claims when a verified source (like a signed JWT) is available. Headers should only be used as a fallback or for metadata, not for core security decisions that conflict with the token.
**Prevention:** Always prioritize verified claims from the JWT payload over request headers. In `require_auth_context`, we now check the token payload first and only look at headers if the information is missing from the token.

## 2024-10-24 - [CRITICAL] JWT Identity Spoofing via Headers
**Vulnerability:** The `require_auth_context` function prioritized `X-Pinak-Client-Id` and `X-Pinak-Client-Name` headers over the authenticated JWT payload claims, allowing any authenticated user to impersonate any client (e.g. admin) by simply adding a header.
**Learning:** This existed because the code used `header or payload` logic (`client_id = header or payload`), effectively trusting the header first.
**Prevention:** Always prioritize trusted claims (JWT) over user-controlled inputs (Headers). The fix inverted the logic to `client_id = payload or header`.

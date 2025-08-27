# Pinak Governance

Pinak-Gov enforces project-scoped isolation and propagates role claims from Bridge tokens to upstream governance services (e.g., Parlant).

Key headers and claims
- `X-Pinak-Project`: required project identity. Must match `pid` claim in the bearer token if provided.
- `Authorization`: JWT signed with `SECRET_KEY`. Must include `pid`. Optional `role` claim.
- `X-Pinak-Role`: forwarded by the gateway from the JWT `role` claim.

Enforcement
- pid==header: Gateway rejects mismatches with 403.
- Role allowlist: Gateway enforces `PINAK_ALLOWED_ROLES` (default: viewer,editor,admin).

Audit
- All mutating requests are mirrored to the memory service `/api/v1/memory/event` as `type=gov_audit`.
- Governance changes appear in changelog as `change_type=governance` with time-range filtering and paging.

Sample project policy

```
// .pinak/policy.json (example)
{
  "rbac": {
    "viewer": ["policy:read", "audit:read"],
    "editor": ["policy:read", "policy:write", "audit:read"],
    "admin": ["*"]
  }
}
```

Roadmap
- Map project policy operations to Parlant authz at the upstream layer.
- Enforce pid==header and role at Parlant for defense-in-depth.

## Role → Operation Mapping (example)

Until upstream enforcement is finalized in Parlant, the gateway forwards `X-Pinak-Role` for contextual policy checks. A typical mapping might look like:

- viewer: read-only operations
  - Parlant: GET /policies, GET /policies/{id}, GET /audit
  - Pinak: GET /pinak-gov/audit
- editor: read + write within project scope
  - Parlant: POST/PUT/PATCH /policies, POST /guidelines
  - Pinak: mirrored to changelog as `change_type=governance`
- admin: full control
  - Parlant: DELETE operations, capability toggles, admin endpoints
  - Pinak: same as above, with heightened audit scrutiny

Recommended practice: define `.pinak/policy.json` per project to document expected operations per role and reference it during reviews.

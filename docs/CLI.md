# Pinak CLI (One-Click)

Use the top-level `pinak` CLI for one-click setup and health checks.

Examples:

```
# One-click: Bridge init + services up + health
pinak quickstart --name "MyApp" --url http://localhost:8011 --tenant default

# Security baseline + environment check
pinak doctor

# Token helper (mints dev JWT and sets via Bridge)
pinak token --set

# Passthrough to sub-CLIs
pinak bridge status --json
pinak memory health

## Token roles and governance

The `pinak token` command now mints a JWT locally with the current project id (pid) from Bridge and an optional role claim. Roles propagate to governance via the `X-Pinak-Role` header and can be restricted by the gateway.

Examples:

```
# Viewer token (default subject: analyst)
pinak token --role viewer --set

# Editor token with custom subject and secret
SECRET_KEY=change-me-in-prod pinak token --sub alice --role editor --set
```

Environment variables influencing governance:

- `PINAK_ALLOWED_ROLES`: comma-separated allowlist enforced by the gateway (default: `viewer,editor,admin`).
- `MEMORY_API_URL`: base URL for the memory API used by the gateway to write audit events (default: `http://memory-api:8000`).
- `PARLANT_IMAGE_REF`: digest-pinned Parlant image reference for compose, e.g. `ghcr.io/org/parlant:1.0.0@sha256:...`.
```

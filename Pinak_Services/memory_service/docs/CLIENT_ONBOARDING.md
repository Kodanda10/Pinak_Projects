# Client Onboarding (Pinak Memory)

## Quick Start

1) Obtain a JWT (admin or agent scope).
2) Register your client once.
3) Use write endpoints with `client_id` and optional child ID.

---

## 1) Register Client

```bash
export PINAK_JWT_SECRET="secret"
TOKEN=$(python -m cli.main mint)

curl -sS -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  http://127.0.0.1:8000/api/v1/memory/client/register \
  -d '{
    "client_id": "codex",
    "client_name": "codex-cli",
    "status": "trusted"
  }'
```

> You can also embed `client_id` and `client_name` into the JWT. Headers are optional and only used for convenience/metadata.

---

## Approval Workflow (Manual)
When a client registers for the first time, its status is `registered`.
Admins should mark the client as **trusted** in the TUI (Clients tab) to enable auto‑approval.

If the MCP client is not trusted, it will show a warning telling the human to approve the registry entry.

---

## 2) Write Memory (Semantic)

```bash
curl -sS -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Pinak-Client-Id: codex" \
  -H "X-Pinak-Client-Name: codex-cli" \
  http://127.0.0.1:8000/api/v1/memory/add \
  -d '{
    "content": "User prefers single-sprint delivery.",
    "tags": ["preference", "delivery"]
  }'
```

### Child Agent (sub‑agent)

```bash
curl -sS -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "X-Pinak-Client-Id: codex" \
  -H "X-Pinak-Child-Client-Id: codex-subagent-1" \
  http://127.0.0.1:8000/api/v1/memory/episodic/add \
  -d '{
    "content": "Summarized the deployment plan.",
    "timestamp": "2026-01-31T12:00:00Z"
  }'
```

---

## 3) Retrieve Context

```bash
curl -sS \
  -H "Authorization: Bearer $TOKEN" \
  "http://127.0.0.1:8000/api/v1/memory/retrieve_context?query=deployment plan"
```

---

## Remote Access (Tailscale / SSH)

### Tailscale (recommended)
1) Install Tailscale on both machines and run:
```bash
tailscale up
```
2) On the server, bind the API to `0.0.0.0` (already configured in the LaunchAgent).
3) On the client, use:
```
http://<tailscale-ip>:8000/api/v1/memory
```

**Current Tailscale IP (as of 2026-02-01):**
```
http://100.66.59.92:8000/api/v1/memory
```

Required headers:
- `Authorization: Bearer <token>`
- `X-Pinak-Client-Id`
- `X-Pinak-Client-Name`
- `X-Pinak-Child-Client-Id` (optional)

### SSH Tunnel
```bash
ssh -L 8000:127.0.0.1:8000 <user>@<pinak-host>
```
Then use `http://127.0.0.1:8000/api/v1/memory` on the client.

---

## Schemas & Templates

Local canonical paths:
- `~/pinak-memory/schemas`
- `~/pinak-memory/templates`

API endpoints:
- `GET /api/v1/memory/schema`
- `GET /api/v1/memory/schema/{layer}`

Doctor syncs these:
```bash
python -m cli.main doctor --fix
```

---

## Lockdown / Unlock (Source Protection)
Keep the repo locked in normal operations:
```bash
sudo scripts/pinak-lockdown.sh
```

If you need to change code:
```bash
sudo scripts/pinak-unlock.sh
# make changes
sudo scripts/pinak-lockdown.sh
```

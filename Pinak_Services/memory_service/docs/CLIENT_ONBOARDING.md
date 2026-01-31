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

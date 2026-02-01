# Observability Schema

## logs_agents
Tracks live agent presence and basic metadata.

Columns:
- `agent_id` (text)
- `client_name` (text)
- `client_id` (text)
- `parent_client_id` (text)
- `hostname` (text, nullable)
- `pid` (text, nullable)
- `status` (text)
- `meta` (json)
- `tenant` (text)
- `project_id` (text)
- `last_seen` (timestamp)

## logs_access
Tracks read/write/propose events for memory access.

Columns:
- `agent_id` (text, nullable)
- `client_name` (text, nullable)
- `client_id` (text, nullable)
- `parent_client_id` (text, nullable)
- `child_client_id` (text, nullable)
- `event_type` (read|write|delete|update|propose)
- `target_layer` (semantic|episodic|procedural|rag|hybrid)
- `query` (text, nullable)
- `memory_id` (text, nullable)
- `result_count` (int, nullable)
- `status` (ok|error)
- `detail` (text, nullable)
- `tenant` (text)
- `project_id` (text)
- `ts` (timestamp)

## memory_quarantine
Holds proposed writes pending approval.

Columns:
- `layer` (semantic|episodic|procedural|rag)
- `payload` (json)
- `status` (pending|approved|rejected)
- `agent_id` (text, nullable)
- `client_id` (text, nullable)
- `client_name` (text, nullable)
- `validation_errors` (json, nullable)
- `tenant` (text)
- `project_id` (text)
- `created_at` (timestamp)
- `reviewed_at` (timestamp, nullable)
- `reviewed_by` (text, nullable)

## logs_client_issues
Captures client ingestion issues, schema errors, and auth anomalies.

Columns:
- `client_id` (text)
- `client_name` (text, nullable)
- `agent_id` (text, nullable)
- `parent_client_id` (text, nullable)
- `child_client_id` (text, nullable)
- `layer` (text, nullable)
- `error_code` (text)
- `message` (text)
- `payload` (json, nullable)
- `metadata` (json, nullable)
- `status` (open|resolved)
- `tenant` (text)
- `project_id` (text)
- `created_at` (timestamp)
- `resolved_at` (timestamp, nullable)
- `resolved_by` (text, nullable)
- `resolution` (text, nullable)

## clients_registry
Tracks observed and registered clients.

Columns:
- `client_id` (text)
- `client_name` (text, nullable)
- `parent_client_id` (text, nullable)
- `status` (observed|registered|trusted|blocked)
- `metadata` (json, nullable)
- `tenant` (text)
- `project_id` (text)
- `created_at` (timestamp)
- `updated_at` (timestamp)
- `last_seen` (timestamp, nullable)

## logs_audit
Tamper-evident audit log (hash chain).

Columns:
- `event_type` (text)
- `payload` (json)
- `prev_hash` (text, nullable)
- `hash` (text)
- `ts` (timestamp)

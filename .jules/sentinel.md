## 2026-02-06 - Mass Assignment Vulnerability in Memory Service
**Vulnerability:** The `update_memory` method allowed updating any field except a small blacklist, enabling attackers to modify immutable fields like `agent_id` or `client_id` via a JSON update request.
**Learning:** Blacklisting fields is fragile because new model fields are automatically exposed unless explicitly added to the blacklist.
**Prevention:** Use strict whitelisting for update operations, defining exactly which fields are mutable for each resource type.

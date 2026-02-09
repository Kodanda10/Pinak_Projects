## 2024-05-22 - Mass Assignment Vulnerability in Memory Updates
**Vulnerability:** The `MemoryService.update_memory` method relied on a blacklist (`forbidden_keys`) to filter updates, which allowed attackers to overwrite sensitive fields like `agent_id` and `client_id` by including them in the payload.
**Learning:** Blacklists are insufficient for input validation as schemas evolve. A whitelist approach (`ALLOWED_UPDATES`) explicitly defines safe fields, preventing unintended modifications.
**Prevention:** Always use strict whitelists for update operations, defining exactly which fields are mutable for each resource type.

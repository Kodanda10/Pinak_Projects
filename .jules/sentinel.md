## 2026-02-11 - Mass Assignment in Update Logic
**Vulnerability:** The `MemoryService.update_memory` method relied on a blacklist (`forbidden_keys`) to sanitize user input before passing it to the database layer. This allowed unauthorized updates to non-blacklisted fields (like `client_id`) and potentially exposed the service to SQL injection if keys were maliciously crafted to include SQL syntax.
**Learning:** Blacklisting fields is fragile because it requires maintenance for every new schema change. It assumes "allow everything else," which violates the principle of least privilege.
**Prevention:** Implement strict whitelisting for update operations. Define exactly which fields are mutable for each resource type and reject everything else. This provides "secure by default" behavior.

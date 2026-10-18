## 2025-02-27 - Appending composite indexes over multi-tenant databases
**Learning:** Adding multi-tenant composite indexes (tenant, project_id) along with status or timestamp fields prevents N+1 queries and full table scans on multi-tenant tables.
**Action:** Always append composite multi-tenant database indexes on heavily queried multi-tenant tables.

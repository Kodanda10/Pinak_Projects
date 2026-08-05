## 2025-08-29 - Missing Indexes for Multi-Tenant Query
**Learning:** Found missing database indexes on `tenant` and `project_id` fields for several multi-tenant queries (working_memory, client_issues, clients_registry, etc)
**Action:** Adding multi-tenant index to `app/core/database.py` tables.

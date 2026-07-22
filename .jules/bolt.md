## 2025-03-01 - Multi-Tenant Query Optimization
**Learning:** Queries in multi-tenant tables (`tenant` and `project_id` required in WHERE clause) suffer performance issues without composite indexes.
**Action:** Always add composite indexes on `(tenant, project_id, [selective_field])` for multi-tenant tables. Ensure mocked schemas in tests are updated to include newly indexed columns to prevent `sqlite3.OperationalError: no such column`.

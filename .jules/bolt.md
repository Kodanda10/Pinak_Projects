## YYYY-MM-DD - [Title]
**Learning:** [Insight]
**Action:** [How to apply next time]
## 2025-02-24 - Missing Composite Indexes
**Learning:** SQLite tables `logs_client_issues` and `memory_quarantine` currently lack composite indexes on `(client_id, tenant, project_id, status)`, which is the exact query pattern used in `count_client_issues` and `count_quarantine`. This causes full table scans on every count.
**Action:** Add these composite indexes in `_init_db()` to optimize these frequent dashboard/polling count queries.

## 2024-05-24 - Missing Composite Indexes
**Learning:** The logs_client_issues and memory_quarantine tables only had simple indexes on status and/or created_at, missing the composite indexes on (client_id, tenant, project_id, status) that optimize multi-column filter queries.
**Action:** Always ensure that multi-column filtering queries (like count_client_issues and count_quarantine) are backed by corresponding composite indexes to prevent full table scans.

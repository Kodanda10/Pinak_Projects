## 2024-05-24 - Database Indexes for Multi-Tenant Queries
**Learning:** In the Pinak Memory Service, older test setups lack 'tenant' columns in schemas, while production queries heavily filter by `tenant` and `project_id`. Without composite indexes on these fields, large multi-tenant databases suffer from O(N) linear scans.
**Action:** When adding new indexes for multi-tenant fields, always conditionally check schema existence using `self._column_exists` to avoid breaking legacy test suites while ensuring production is optimized.

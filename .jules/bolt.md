## 2024-10-24 - Missing Database Indexes on Frequently Queried Fields
**Learning:** The database schemas were frequently queried for specific tenants and projects across all data sets, without any underlying composite indexes, resulting in unoptimized full table scans (O(N)).
**Action:** Always add composite indexes targeting the WHERE clauses in primary queries to achieve O(log N) lookup time for frequent filtering scenarios.

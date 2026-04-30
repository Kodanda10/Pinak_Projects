## 2024-05-18 - Prevent SQLite Full Table Scans for Client Layer Stats
**Learning:** Adding columns through `ALTER TABLE ADD COLUMN` doesn't automatically create indexes. When building functions like `get_client_layer_stats` that query by `(tenant, project_id, client_id)`, full table scans occur across all memory layers unless composite indexes are explicitly added.
**Action:** When creating or adding columns used in recurring `WHERE` or `GROUP BY` clauses, always explicitly append `CREATE INDEX` statements to the `_init_db` setup block to ensure efficient queries.

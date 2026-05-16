## 2024-05-16 - Add Composite Index for embedding_ids Retrieval
**Learning:** The `get_memories_by_embedding_ids` query was lacking indexing. `get_memories_by_embedding_ids` is frequently called during Vector Search result resolution. It runs:
`SELECT *, '{mtype}' as type FROM {table} WHERE embedding_id IN ({placeholders}) AND tenant = ? AND project_id = ?`
Since `embedding_id` might be unique, or combined with `tenant` and `project_id`, querying it without an index scans the whole table.
**Action:** Always add indexes for fields that are frequently queried, especially in the core retrieval flow like `get_memories_by_embedding_ids`.

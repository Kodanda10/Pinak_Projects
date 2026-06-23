import sqlite3
import os

db_path = "test.db"
if os.path.exists(db_path):
    os.remove(db_path)

conn = sqlite3.connect(db_path)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS memories_semantic (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    tags TEXT, -- JSON list
    embedding_id INTEGER, -- Link to FAISS
    agent_id TEXT,
    client_id TEXT,
    client_name TEXT,
    tenant TEXT NOT NULL,
    project_id TEXT NOT NULL,
    created_at TEXT NOT NULL
);
""")

cur.execute("EXPLAIN QUERY PLAN SELECT * FROM memories_semantic WHERE embedding_id IN (1, 2) AND tenant = 't1' AND project_id = 'p1'")
for row in cur.fetchall():
    print(row)

cur.execute("""
CREATE INDEX idx_memories_semantic_embedding ON memories_semantic (embedding_id, tenant, project_id);
""")

print("--- After Index ---")
cur.execute("EXPLAIN QUERY PLAN SELECT * FROM memories_semantic WHERE embedding_id IN (1, 2) AND tenant = 't1' AND project_id = 'p1'")
for row in cur.fetchall():
    print(row)

conn.close()
os.remove(db_path)

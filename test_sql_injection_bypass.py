import sqlite3

db = sqlite3.connect(':memory:')
db.execute("CREATE TABLE test (id TEXT PRIMARY KEY, tenant TEXT, project_id TEXT, secret TEXT)")
db.execute("INSERT INTO test (id, tenant, project_id, secret) VALUES ('1', 't1', 'p1', 'safe')")
db.execute("INSERT INTO test (id, tenant, project_id, secret) VALUES ('2', 't2', 'p2', 'safe2')")

updates = {"secret": "hacked", "id = '2' --": "ignored"}
serialized = updates
set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])
query = f"UPDATE test SET {set_clause} WHERE id = ? AND tenant = ? AND project_id = ?"
params = list(serialized.values()) + ['1', 't1', 'p1']

print(f"Query: {query}")
print(f"Params: {params}")
try:
    db.execute(query, params)
    print("Execution successful")
except Exception as e:
    print(f"Error: {e}")

cur = db.execute("SELECT * FROM test")
print(cur.fetchall())

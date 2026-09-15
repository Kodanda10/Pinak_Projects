import sqlite3
db = sqlite3.connect(':memory:')
db.execute("CREATE TABLE test (id TEXT PRIMARY KEY, tenant TEXT, project_id TEXT, secret TEXT)")
db.execute("INSERT INTO test (id, tenant, project_id, secret) VALUES ('1', 't1', 'p1', 'safe')")
db.execute("INSERT INTO test (id, tenant, project_id, secret) VALUES ('2', 't2', 'p2', 'safe2')")

# The original logic:
# safe_updates = {k: v for k, v in updates.items() if k not in forbidden_keys}
updates = {"secret = (SELECT 'hacked')": "value"}
set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
query = f"UPDATE test SET {set_clause} WHERE id = ? AND tenant = ? AND project_id = ?"

db.execute(query, list(updates.values()) + ['2', 't2', 'p2'])
cur = db.execute("SELECT * FROM test")
print(cur.fetchall())

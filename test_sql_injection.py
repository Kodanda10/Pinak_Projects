import sqlite3

db = sqlite3.connect(':memory:')
db.execute("CREATE TABLE test (id TEXT PRIMARY KEY, value TEXT)")
db.execute("INSERT INTO test (id, value) VALUES ('1', 'abc')")

# SQL Injection scenario
import json
memory_id = "1"
tenant = "t1"
project_id = "p1"
serialized = {"value": "def"}

# The vulnerable set clause creation
set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])
print(f"Set clause: {set_clause}")

# Trying injection on column name
serialized = {"value": "def", "invalid_col = 1; DROP TABLE test; --": "xyz"}
set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])
print(f"Set clause with injection: {set_clause}")

query = f"UPDATE test SET {set_clause} WHERE id = ?"
params = list(serialized.values()) + [memory_id]
print(f"Query: {query}")
print(f"Params: {params}")

try:
    db.execute(query, params)
    print("Injection successful (or executed without syntax error)")
except sqlite3.OperationalError as e:
    print(f"Error: {e}")

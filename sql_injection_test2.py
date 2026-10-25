import json

updates = {
    "content": "new content",
    "invalid-col": "value",
    "id": "123", # protected
    "tenant": "t1", # protected
    "project_id": "p1", # protected
    "created_at": "now", # protected
    "tags": ["a", "b"]
}

ALLOWED_COLUMNS = {
    "content", "tags", "embedding_id", "agent_id", "client_id", "client_name",
    "goal", "outcome", "plan", "steps", "salience",
    "skill_name", "trigger", "description", "code_snippet",
    "query", "external_source"
}

serialized = {}
for key, value in updates.items():
    if key not in ALLOWED_COLUMNS or not key.isidentifier():
        continue
    if key in ("tags", "plan", "steps") and value is not None:
        serialized[key] = json.dumps(value)
    else:
        serialized[key] = value

if not serialized:
    print("No valid updates")
else:
    set_clause = ", ".join([f"{k} = ?" for k in serialized.keys()])
    params = list(serialized.values()) + ["id1", "t1", "p1"]
    print(f"set_clause: {set_clause}")
    print(f"params: {params}")

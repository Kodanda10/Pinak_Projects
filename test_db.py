with open("Pinak_Services/memory_service/app/core/database.py", "r") as f:
    content = f.read()

# verify literal replacement strings
old_code_1 = """            conn.execute(\"\"\"
                CREATE INDEX IF NOT EXISTS idx_clients_registry_status
                ON clients_registry (status);
            \"\"\")"""

print("Match 1:", old_code_1 in content)

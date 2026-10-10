with open("Pinak_Services/memory_service/app/core/database.py", "r") as f:
    content = f.read()

new_content = content.replace("""            conn.execute(\"\"\"
                CREATE INDEX IF NOT EXISTS idx_clients_registry_status
                ON clients_registry (status);
            \"\"\")""", """            conn.execute(\"\"\"
                CREATE INDEX IF NOT EXISTS idx_clients_registry_status
                ON clients_registry (status);
            \"\"\")
            conn.execute(\"\"\"
                CREATE INDEX IF NOT EXISTS idx_clients_registry_tenant_project
                ON clients_registry (tenant, project_id);
            \"\"\")""")

new_content = new_content.replace("""            conn.execute(\"\"\"
                CREATE INDEX IF NOT EXISTS idx_logs_agents_last_seen
                ON logs_agents (last_seen);
            \"\"\")""", """            conn.execute(\"\"\"
                CREATE INDEX IF NOT EXISTS idx_logs_agents_last_seen
                ON logs_agents (last_seen);
            \"\"\")
            conn.execute(\"\"\"
                CREATE INDEX IF NOT EXISTS idx_logs_agents_tenant_project
                ON logs_agents (tenant, project_id, last_seen DESC);
            \"\"\")""")

new_content = new_content.replace("""            conn.execute(\"\"\"
                CREATE INDEX IF NOT EXISTS idx_logs_access_ts
                ON logs_access (ts);
            \"\"\")""", """            conn.execute(\"\"\"
                CREATE INDEX IF NOT EXISTS idx_logs_access_ts
                ON logs_access (ts);
            \"\"\")
            conn.execute(\"\"\"
                CREATE INDEX IF NOT EXISTS idx_logs_access_tenant_project_ts
                ON logs_access (tenant, project_id, ts DESC);
            \"\"\")""")

new_content = new_content.replace("""            conn.execute(\"\"\"
                CREATE INDEX IF NOT EXISTS idx_memory_quarantine_status
                ON memory_quarantine (status);
            \"\"\")""", """            conn.execute(\"\"\"
                CREATE INDEX IF NOT EXISTS idx_memory_quarantine_status
                ON memory_quarantine (status);
            \"\"\")
            conn.execute(\"\"\"
                CREATE INDEX IF NOT EXISTS idx_memory_quarantine_tenant_project_status
                ON memory_quarantine (tenant, project_id, status);
            \"\"\")""")

new_content = new_content.replace("""            conn.execute(\"\"\"
                CREATE INDEX IF NOT EXISTS idx_logs_client_issues_ts
                ON logs_client_issues (created_at);
            \"\"\")""", """            conn.execute(\"\"\"
                CREATE INDEX IF NOT EXISTS idx_logs_client_issues_ts
                ON logs_client_issues (created_at);
            \"\"\")
            conn.execute(\"\"\"
                CREATE INDEX IF NOT EXISTS idx_logs_client_issues_tenant_project_status
                ON logs_client_issues (tenant, project_id, status);
            \"\"\")""")

if new_content != content:
    with open("Pinak_Services/memory_service/app/core/database.py", "w") as f:
        f.write(new_content)
    print("Database manager updated successfully!")
else:
    print("No changes made.")

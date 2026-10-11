import sqlite3

def patch_db():
    with open('app/core/database.py', 'r') as f:
        content = f.read()

    new_code = """
            self._ensure_column(conn, "working_memory", "expires_at", "TEXT")

            # ⚡ BOLT OPTIMIZATION: Composite multi-tenant indexes
            # Prevents full table scans on heavily queried multi-tenant tables.
            # Adding composite indexes for (tenant, project_id) along with frequently filtered/sorted columns.
            # Expected impact: Multi-tenant queries become O(log N) instead of O(N), significantly faster.
            if self._column_exists(conn, "logs_access", "tenant"):
                conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_access_tenant_ts ON logs_access (tenant, project_id, ts);")
            if self._column_exists(conn, "memory_quarantine", "tenant"):
                conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_q_tenant_status ON memory_quarantine (tenant, project_id, status);")
            if self._column_exists(conn, "clients_registry", "tenant"):
                conn.execute("CREATE INDEX IF NOT EXISTS idx_clients_reg_tenant_status ON clients_registry (tenant, project_id, status);")
            if self._column_exists(conn, "logs_agents", "tenant"):
                conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_agents_tenant_seen ON logs_agents (tenant, project_id, last_seen);")
            if self._column_exists(conn, "logs_client_issues", "tenant"):
                conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_client_iss_tenant_status ON logs_client_issues (tenant, project_id, status);")
"""

    old_code = """
            self._ensure_column(conn, "working_memory", "expires_at", "TEXT")"""

    if old_code in content:
        content = content.replace(old_code, new_code)
        with open('app/core/database.py', 'w') as f:
            f.write(content)
        print("Patched database.py")
    else:
        print("Failed to find old code")

patch_db()

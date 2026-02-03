from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Static, DataTable, Label, Button
from textual.reactive import reactive
from textual.screen import Screen
from textual.message import Message
from typing import Optional, Dict, Tuple, Any
import sqlite3
import os
import datetime
import json
import numpy as np
import subprocess

# --- CSS with Visual Polish ---
GLOBAL_CSS = """
Screen {
    background: #05070D;
    color: #E5E7EB;
}

Header {
    background: #0B1220;
    color: #E5E7EB;
    dock: top;
    text-style: bold;
}

Footer {
    background: #0B1220;
    color: #9CA3AF;
    dock: bottom;
}

#sidebar {
    dock: left;
    width: 36;
    background: #0A0F1A;
    border-right: solid #111827;
}

#sidebar .nav-title {
    padding: 1 2;
    color: #64748B;
    text-style: bold;
}

.nav-list {
    height: 100%;
}

Button.nav-button {
    margin: 0 1;
    padding: 1 2;
    width: 1fr;
    background: #0A0F1A;
    color: #A3B0C2;
    border: none;
    text-align: left;
    text-wrap: nowrap;
    overflow: hidden;
}

Button.nav-button:hover {
    background: #0E1628;
    color: #E5E7EB;
}

Button.nav-button.--highlight {
    background: #F97316;
    color: #0B0F17;
    text-style: bold;
    border-left: solid #FDBA74;
}

#main-content {
    padding: 1 2;
    height: 100%;
    background: #0B111E;
}

#topbar {
    height: 8;
    padding: 1 3;
    background: #0F172A;
    border: solid #1E293B;
    margin: 0 0 1 0;
}

#topbar-spacer {
    width: 28;
}

#topbar-center {
    width: 1fr;
    align: center middle;
}

#topbar-title {
    color: #FBBF24;
    text-style: bold;
    text-align: center;
    padding-top: 1;
}

#topbar-subtitle {
    color: #A5B4D0;
    text-align: center;
    padding-bottom: 1;
}

#topbar-status {
    width: 22;
    background: #0C2233;
    border: round #334155;
    color: #93C5FD;
    text-style: dim;
    text-align: center;
    padding: 0 2;
}

.stat-card {
    background: #0F172A;
    border: solid #1E293B;
    height: 7;
    margin: 1;
    padding: 0 1;
    width: 1fr;
    border-title-color: #F59E0B;
    border-title-style: bold;
}

.stat-title {
    color: #FDBA74;
    text-style: bold;
    border-bottom: solid #1E293B;
    margin-bottom: 1;
}

.stat-value {
    text-align: center;
    color: #22D3EE;
    text-style: bold;
    margin-top: 1;
}

.panel {
    background: #0F172A;
    border: solid #1E293B;
    padding: 1 2;
}

DataTable {
    background: #0F172A;
    border: solid #1E293B;
    color: #E5E7EB;
}

.section-header {
    color: #F59E0B;
    text-style: bold;
    border-bottom: solid #F59E0B;
    margin-bottom: 1;
}

Button {
    background: #111827;
    color: #E5E7EB;
    border: solid #1E293B;
}

Button.-primary {
    background: #F97316;
    color: #0B0F17;
    border: solid #FDBA74;
    text-style: bold;
}

.toolbar {
    height: 3;
    margin: 0 0 1 0;
    align: left middle;
}

.toolbar Button {
    margin-right: 1;
}

#toolbar-spacer {
    width: 1fr;
}

#client-selected {
    padding-right: 1;
    padding-left: 1;
    color: #A5B4D0;
    text-align: right;
    width: 28;
}

.muted {
    color: #94A3B8;
}

.mono {
    color: #A3B0C2;
}
"""

IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

def _parse_ts(value: Optional[str]) -> Optional[datetime.datetime]:
    if not value:
        return None
    s = value.strip()
    try:
        if "T" in s:
            if s.endswith("Z"):
                s = s.replace("Z", "+00:00")
            dt = datetime.datetime.fromisoformat(s)
        else:
            dt = datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            dt = dt.replace(tzinfo=datetime.timezone.utc)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt

def _format_ts(value: Optional[str], with_date: bool = False) -> str:
    dt = _parse_ts(value)
    if not dt:
        return value or ""
    dt_ist = dt.astimezone(IST)
    return dt_ist.strftime("%m-%d %H:%M") if with_date else dt_ist.strftime("%H:%M:%S")

class Sidebar(Container):
    def compose(self) -> ComposeResult:
        yield Label("CORE", classes="nav-title")
        with Vertical(classes="nav-list"):
            yield Button("📊 System Mesh", id="nav-dashboard", classes="nav-button")
            yield Button("📡 Memory Access", id="nav-access", classes="nav-button")
            yield Button("👥 Agent Swarm", id="nav-agents", classes="nav-button")
            yield Button("🧷 Client Issues", id="nav-issues", classes="nav-button")
            yield Button("🧭 Client Registry", id="nav-clients", classes="nav-button")
            yield Button("❤️ Bio-Health", id="nav-health", classes="nav-button")

class DashboardView(Container):
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Container(
                Static("Synaptic Memories", classes="stat-title"),
                Static("0", id="stat-total-mem", classes="stat-value"),
                classes="stat-card"
            )
            yield Container(
                Static("Active Agents", classes="stat-title"),
                Static("0", id="stat-active-agents", classes="stat-value"),
                classes="stat-card"
            )
            yield Container(
                Static("Access / Min", classes="stat-title"),
                Static("0", id="stat-access-rate", classes="stat-value"),
                classes="stat-card"
            )
        with Horizontal():
             yield Container(
                Static("Substrate Volume", classes="stat-title"),
                Static("0.00 MB", id="stat-db-size", classes="stat-value"),
                classes="stat-card"
            )
             yield Container(
                Static("Vector Capacity", classes="stat-title"),
                Static("0", id="stat-vec-size", classes="stat-value"),
                classes="stat-card"
            )
             yield Container(
                Static("Ingest / Min", classes="stat-title"),
                Static("0", id="stat-ingest-rate", classes="stat-value"),
                classes="stat-card"
            )

    def on_mount(self):
        self.set_interval(2, self.refresh_stats)
        self.refresh_stats()

    def refresh_stats(self):
        db_path = "data/memory.db"
        vec_path = "data/vectors.index.npy"

        total_mem = 0
        active_agents = 0
        db_size = "0 MB"
        vec_count = 0
        access_rate = 0
        ingest_rate = 0

        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                count = 0
                for t in ["memories_semantic", "memories_episodic", "memories_procedural", "working_memory"]:
                    cur.execute(f"SELECT count(*) FROM {t}")
                    count += cur.fetchone()[0]
                total_mem = count
                # Active agents in last 60s
                cur.execute("SELECT count(*) FROM logs_agents WHERE last_seen >= datetime('now','-60 seconds')")
                active_agents = cur.fetchone()[0]
                conn.close()
                size_mb = os.path.getsize(db_path) / (1024*1024)
                db_size = f"{size_mb:.2f} MB"
            except: pass

        if os.path.exists(vec_path):
            try:
                index_data = np.load(vec_path, allow_pickle=True)
                if hasattr(index_data, 'item') and isinstance(index_data.item(), dict):
                    vec_count = len(index_data.item().get('ids', []))
                else:
                    vec_count = index_data.shape[0]
            except: vec_count = -1

        # Access + ingest rates (last 60s)
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute("SELECT count(*) FROM logs_access WHERE ts >= datetime('now','-60 seconds')")
                access_rate = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM logs_access WHERE event_type = 'write' AND ts >= datetime('now','-60 seconds')")
                ingest_rate = cur.fetchone()[0]
                conn.close()
            except: pass

        self.query_one("#stat-total-mem", Static).update(str(total_mem))
        self.query_one("#stat-active-agents", Static).update(str(active_agents))
        self.query_one("#stat-db-size", Static).update(db_size)
        self.query_one("#stat-vec-size", Static).update(str(vec_count))
        self.query_one("#stat-access-rate", Static).update(str(access_rate))
        self.query_one("#stat-ingest-rate", Static).update(str(ingest_rate))

class AccessView(Container):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._access_rows: Dict[str, Dict[str, Any]] = {}

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Memory Access Stream", classes="stat-title")
            yield DataTable(id="access-table")
        with Vertical(id="access-detail", classes="panel"):
            yield Static("Access Detail", classes="stat-title")
            yield VerticalScroll(Static("Select a row to view memory details.", id="access-detail-text"))

    def on_mount(self):
        table = self.query_one("#access-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Time", "Client", "Agent", "Type", "Layer", "Query/ID", "Status")
        table.zebra_stripes = True
        self.set_interval(2, self.refresh_access)
        self.refresh_access()

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        if event.data_table.id != "access-table":
            return
        row_key = str(event.row_key)
        row = self._access_rows.get(row_key)
        if row:
            self.show_access_detail(row)

    def refresh_access(self):
        db_path = "data/memory.db"
        if not os.path.exists(db_path):
            return
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT id, ts, client_name, client_id, agent_id, event_type, target_layer, query, memory_id, result_count, status, detail
                FROM logs_access
                ORDER BY ts DESC
                LIMIT 50
            """)
            rows = cur.fetchall()
            conn.close()

            table = self.query_one("#access-table", DataTable)
            table.clear()
            self._access_rows = {}
            for row in rows:
                row_id = row["id"]
                self._access_rows[str(row_id)] = dict(row)
                ts = _format_ts(row["ts"])
                client = row["client_name"] or row["client_id"] or "unknown"
                agent = row["agent_id"] or "unknown"
                etype = row["event_type"]
                layer = row["target_layer"] or "-"
                payload = row["query"] or row["memory_id"] or "-"
                status = row["status"]
                if etype == "write":
                    etype = "✳ write"
                elif etype == "read":
                    etype = "• read"
                elif etype == "propose":
                    etype = "◇ propose"
                table.add_row(ts, client, agent, etype, layer, payload[:40], status, key=str(row_id))
        except Exception:
            return

    def _format_jsonish(self, value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=True, indent=2)
        if isinstance(value, str):
            s = value.strip()
            if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                try:
                    parsed = json.loads(s)
                    return json.dumps(parsed, ensure_ascii=True, indent=2)
                except Exception:
                    return value
        return str(value)

    def _load_memory_row(self, table: str, memory_id: str) -> Optional[Dict[str, Any]]:
        db_path = "data/memory.db"
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute(f"SELECT * FROM {table} WHERE id = ?", (memory_id,))
                row = cur.fetchone()
            if not row:
                return None
            return dict(row)
        except Exception:
            return None

    def _load_quarantine_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        db_path = "data/memory.db"
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT * FROM memory_quarantine
                    WHERE layer = ?
                      AND (client_id = ? OR client_name = ? OR agent_id = ?)
                    ORDER BY ABS(strftime('%s', created_at) - strftime('%s', ?)) ASC
                    LIMIT 1
                    """,
                    (
                        row.get("target_layer"),
                        row.get("client_id") or "",
                        row.get("client_name") or "",
                        row.get("agent_id") or "",
                        row.get("ts") or "",
                    ),
                )
                item = cur.fetchone()
            if not item:
                return None
            return dict(item)
        except Exception:
            return None

    def show_access_detail(self, row: Dict[str, Any]) -> None:
        lines = []
        lines.append(f"[bold #F8B24B]Access Detail[/bold #F8B24B]")
        lines.append(f"[#9CA3AF]Time:[/#9CA3AF] {_format_ts(row.get('ts'), with_date=True)}")
        lines.append(f"[#9CA3AF]Client:[/#9CA3AF] {row.get('client_name') or row.get('client_id') or '-'}")
        lines.append(f"[#9CA3AF]Agent:[/#9CA3AF] {row.get('agent_id') or '-'}")
        lines.append(f"[#9CA3AF]Type:[/#9CA3AF] {row.get('event_type') or '-'}")
        lines.append(f"[#9CA3AF]Layer:[/#9CA3AF] {row.get('target_layer') or '-'}")
        lines.append(f"[#9CA3AF]Status:[/#9CA3AF] {row.get('status') or '-'}")
        if row.get("query"):
            lines.append(f"[#9CA3AF]Query:[/#9CA3AF] {row.get('query')}")
        if row.get("detail"):
            lines.append(f"[#9CA3AF]Detail:[/#9CA3AF] {row.get('detail')}")

        memory_id = row.get("memory_id")
        layer = row.get("target_layer") or ""
        table_map = {
            "semantic": "memories_semantic",
            "episodic": "memories_episodic",
            "procedural": "memories_procedural",
            "rag": "memories_rag",
            "working": "working_memory",
        }

        if row.get("event_type") == "propose":
            item = self._load_quarantine_row(row)
            if item:
                lines.append("")
                lines.append("[bold #F59E0B]Quarantine Payload[/bold #F59E0B]")
                lines.append(self._format_jsonish(item.get("payload")))
            else:
                lines.append("")
                lines.append("[#9CA3AF]No quarantine payload matched.[/#9CA3AF]")
        elif memory_id and layer in table_map:
            mem = self._load_memory_row(table_map[layer], memory_id)
            if mem:
                lines.append("")
                lines.append("[bold #F59E0B]Memory Record[/bold #F59E0B]")
                for key in ["content", "goal", "outcome", "skill_name", "steps", "tags", "plan", "tool_logs", "metadata"]:
                    if key in mem and mem.get(key) is not None:
                        lines.append(f"[#9CA3AF]{key}:[/#9CA3AF] {self._format_jsonish(mem.get(key))}")
            else:
                lines.append("")
                lines.append("[#9CA3AF]Memory record not found.[/#9CA3AF]")
        else:
            lines.append("")
            lines.append("[#9CA3AF]No memory payload for this access event.[/#9CA3AF]")

        self.query_one("#access-detail-text", Static).update("\n".join(lines))

class AgentsView(Container):
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Agent Swarm - Live Presence", classes="stat-title")
            yield DataTable(id="agents-table")
        with Vertical(id="agent-details", classes="stat-card"):
            yield Static("Entity Cognition / History", classes="stat-title")
            yield VerticalScroll(Static("Select an entity to view history", id="agent-summary-text"))

    def on_mount(self):
        table = self.query_one("#agents-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Status", "Client", "Agent", "Last Seen", "Reads(5m)", "Writes(5m)", "Errors(5m)")
        table.zebra_stripes = True
        self.refresh_agents()
        self.set_interval(5, self.refresh_agents)

    def refresh_agents(self):
        db_path = "data/memory.db"
        if not os.path.exists(db_path): return
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            cur.execute("""
                SELECT agent_id, client_name, status, last_seen
                FROM logs_agents
                ORDER BY last_seen DESC
            """)
            agents = cur.fetchall()

            cur.execute("""
                SELECT agent_id,
                       SUM(CASE WHEN event_type = 'read' AND ts >= datetime('now','-5 minutes') THEN 1 ELSE 0 END) AS reads_5m,
                       SUM(CASE WHEN event_type = 'write' AND ts >= datetime('now','-5 minutes') THEN 1 ELSE 0 END) AS writes_5m,
                       SUM(CASE WHEN status = 'error' AND ts >= datetime('now','-5 minutes') THEN 1 ELSE 0 END) AS errors_5m
                FROM logs_access
                GROUP BY agent_id
            """)
            access_counts = {row[0]: row[1:] for row in cur.fetchall()}
            conn.close()

            table = self.query_one("#agents-table", DataTable)
            table.clear()
            now = datetime.datetime.now(datetime.timezone.utc)
            for agent_id, client_name, status, last_seen in agents:
                last_seen_ts = _parse_ts(last_seen)
                delta = (now - last_seen_ts).total_seconds() if last_seen_ts else 9999
                if delta <= 30:
                    indicator = "🟢"
                elif delta <= 120:
                    indicator = "🟡"
                else:
                    indicator = "⚫"
                counts = access_counts.get(agent_id, (0, 0, 0))
                reads_5m, writes_5m, errors_5m = counts
                table.add_row(
                    indicator,
                    client_name or "unknown",
                    agent_id or "unknown",
                    _format_ts(last_seen),
                    str(reads_5m),
                    str(writes_5m),
                    str(errors_5m),
                )
        except: pass

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        if event.data_table.id == "agents-table":
            agent_id = event.data_table.get_row_at(event.row_key)[2]
            self.show_agent_history(agent_id)

    def show_agent_history(self, agent_id):
        db_path = "data/memory.db"
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT event_type, target_layer, query, memory_id, status, ts
                FROM logs_access
                WHERE agent_id = ?
                ORDER BY ts DESC LIMIT 12
            """, (agent_id,))
            rows = cur.fetchall()
            conn.close()

            text = f"[bold azure]Recent Activity for {agent_id}[/bold azure]\n\n"
            if not rows:
                text += "[yellow]No recent access events.[/yellow]"
            for r in rows:
                ts = _format_ts(r['ts'])
                detail = r['query'] or r['memory_id'] or "-"
                text += f"[bold #60A5FA]◈ {ts} | {r['event_type']} {r['target_layer']}[/bold #60A5FA]\n"
                text += f"[#9CA3AF]{detail} ({r['status']})[/#9CA3AF]\n\n"

            self.query_one("#agent-summary-text", Static).update(text)
        except Exception as e:
            self.query_one("#agent-summary-text", Static).update(f"Error: {e}")

class ClientIssuesView(Container):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.status_filter = "all"

    def compose(self) -> ComposeResult:
        yield Static("Client Issues (Ingestion / Schema / Auth)", classes="stat-title")
        with Horizontal(classes="toolbar"):
            yield Button("Open", id="btn_issues_open", classes="-primary")
            yield Button("All", id="btn_issues_all")
            yield Button("Resolved", id="btn_issues_resolved")
            yield Static("", id="toolbar-spacer")
            yield Static("Filter: all", id="issues-filter", classes="muted")
        yield DataTable(id="issues-table")

    def on_mount(self):
        table = self.query_one("#issues-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Time", "Status", "Client", "Agent", "Layer", "Code", "Message")
        table.zebra_stripes = True
        self.set_interval(3, self.refresh_issues)
        self.refresh_issues()

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn_issues_open":
            self.status_filter = "open"
        elif event.button.id == "btn_issues_resolved":
            self.status_filter = "resolved"
        elif event.button.id == "btn_issues_all":
            self.status_filter = "all"
        self.query_one("#issues-filter", Static).update(f"Filter: {self.status_filter}")
        self.refresh_issues()

    def refresh_issues(self):
        db_path = "data/memory.db"
        if not os.path.exists(db_path):
            return
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            if self.status_filter == "all":
                cur.execute("""
                    SELECT id, created_at, status, client_id, client_name, agent_id, layer, error_code, message
                    FROM logs_client_issues
                    ORDER BY created_at DESC
                    LIMIT 80
                """)
            else:
                cur.execute(
                    """
                    SELECT id, created_at, status, client_id, client_name, agent_id, layer, error_code, message
                    FROM logs_client_issues
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT 80
                    """,
                    (self.status_filter,),
                )
            rows = cur.fetchall()
            conn.close()

            table = self.query_one("#issues-table", DataTable)
            table.clear()
            for row in rows:
                ts = _format_ts(row["created_at"])
                status = row["status"] or "open"
                status_label = "open" if status == "open" else "resolved"
                client = row["client_id"] or row["client_name"] or "unknown"
                agent = row["agent_id"] or "-"
                layer = row["layer"] or "-"
                code = row["error_code"] or "-"
                msg = (row["message"] or "-")[:80]
                table.add_row(ts, status_label, client, agent, layer, code, msg)
        except Exception:
            return


class ClientRegistryView(Container):
    selected_client_id: Optional[str] = None

    def compose(self) -> ComposeResult:
        yield Static("Client Registry (Observed / Registered / Trusted)", classes="stat-title")
        with Horizontal(classes="toolbar"):
            yield Button("Mark Trusted", id="btn_client_trusted", classes="-primary")
            yield Button("Mark Observed", id="btn_client_observed")
            yield Button("Mark Blocked", id="btn_client_blocked")
            yield Static("", id="toolbar-spacer")
            yield Static("Selected: -", id="client-selected", classes="muted")
        yield DataTable(id="clients-table")

    def on_mount(self):
        table = self.query_one("#clients-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Last Seen", "Status", "Client ID", "Name", "Parent", "Updated")
        table.zebra_stripes = True
        table.cursor_type = "row"
        table.focus()
        self.set_interval(5, self.refresh_clients)
        self.refresh_clients()

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        if event.data_table.id != "clients-table":
            return
        self.selected_client_id = str(event.row_key)
        self.query_one("#client-selected", Static).update(f"Selected: {self.selected_client_id}")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn_client_trusted":
            self._update_client_status("trusted")
        elif event.button.id == "btn_client_observed":
            self._update_client_status("observed")
        elif event.button.id == "btn_client_blocked":
            self._update_client_status("blocked")

    def _update_client_status(self, status: str):
        if not self.selected_client_id:
            return
        db_path = "data/memory.db"
        if not os.path.exists(db_path):
            return
        try:
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    "UPDATE clients_registry SET status = ?, updated_at = datetime('now') WHERE client_id = ?",
                    (status, self.selected_client_id),
                )
                conn.commit()
            self.refresh_clients()
        except Exception:
            return

    def refresh_clients(self):
        db_path = "data/memory.db"
        if not os.path.exists(db_path):
            return
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""
                SELECT client_id, client_name, parent_client_id, status, updated_at, last_seen
                FROM clients_registry
                ORDER BY (last_seen IS NULL), last_seen DESC, updated_at DESC
                LIMIT 120
            """)
            rows = cur.fetchall()
            conn.close()

            table = self.query_one("#clients-table", DataTable)
            table.clear()
            for row in rows:
                last_seen = _format_ts(row["last_seen"])
                updated = _format_ts(row["updated_at"])
                row_key = row["client_id"] or row["client_name"] or f"row-{table.row_count}"
                table.add_row(
                    last_seen,
                    row["status"] or "observed",
                    row["client_id"] or "unknown",
                    row["client_name"] or "-",
                    row["parent_client_id"] or "-",
                    updated,
                    key=row_key,
                )
        except Exception:
            return

class HealthView(Container):
    def compose(self) -> ComposeResult:
        yield Static("Substrate Self-Diagnostic", classes="stat-title")
        yield VerticalScroll(Static("Scanning bio-signatures...", id="health_report"))
        yield Button("Run Doctor", id="btn_doctor", variant="primary")
        yield Static("LaunchAgent Health", classes="stat-title")
        yield DataTable(id="launch-table")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn_doctor": self.run_doctor(allow_heavy=True)

    def on_mount(self):
        self.run_doctor(allow_heavy=True)
        table = self.query_one("#launch-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Label", "State", "PID", "Notes")
        table.zebra_stripes = True
        self.set_interval(10, self.refresh_launch_agents)
        self.refresh_launch_agents()

    def run_doctor(self, allow_heavy: bool = False):
        from cli.doctor import run_doctor

        report = run_doctor(fix=True, allow_heavy=allow_heavy)
        lines = []

        if report.actions:
            lines.append("[bold green]🛠 Fixes Applied[/bold green]")
            lines.extend([f"[green]• {a}[/green]" for a in report.actions])

        if report.issues:
            lines.append("[bold red]⚠ Issues Found[/bold red]")
            lines.extend([f"[red]• {i}[/red]" for i in report.issues])
        else:
            lines.append("[green]✅ All Systems Operational[/green]")

        if report.notes:
            lines.append("[bold cyan]ℹ Notes[/bold cyan]")
            lines.extend([f"[cyan]• {n}[/cyan]" for n in report.notes])

        self.query_one("#health_report", Static).update("\n".join(lines))

    def refresh_launch_agents(self):
        table = self.query_one("#launch-table", DataTable)
        table.clear()
        uid = os.getuid()
        labels = [
            ("Server", "com.pinak.memory.server"),
            ("Watchdog", "com.pinak.memory.watchdog"),
            ("Doctor", "com.pinak.memory.doctor"),
            ("Backup", "com.pinak.memory.backup"),
            ("TUI", "com.pinak.memory.tui"),
        ]
        for title, label in labels:
            state = "unloaded"
            pid = "-"
            notes = ""
            try:
                result = subprocess.run(
                    ["/bin/launchctl", "print", f"gui/{uid}/{label}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0:
                    out = result.stdout or ""
                    if "state = running" in out:
                        state = "running"
                    else:
                        state = "loaded"
                    for line in out.splitlines():
                        if line.strip().startswith("pid ="):
                            pid = line.split("=", 1)[-1].strip()
                            break
                else:
                    notes = (result.stderr or result.stdout or "").strip()[:80]
            except Exception as exc:
                notes = str(exc)[:80]
            table.add_row(f"{title} ({label})", state, pid, notes or "-")

class MemoryApp(App):
    CSS = GLOBAL_CSS
    TAB_META: Dict[str, Tuple[str, str]] = {
        "dashboard": ("System Mesh", "Global signals and memory substrate"),
        "access": ("Memory Access", "Read/write stream across layers"),
        "agents": ("Agent Swarm", "Live presence and activity deltas"),
        "issues": ("Client Issues", "Ingestion, schema, and auth anomalies"),
        "clients": ("Client Registry", "Observed, registered, trusted"),
        "health": ("Bio-Health", "Doctor status and launch agents"),
    }
    BINDINGS = [
        ("q", "quit", "Shutdown"),
        ("1", "show_dashboard", "Mesh"),
        ("2", "show_access", "Access"),
        ("3", "show_agents", "Agents"),
        ("4", "show_issues", "Issues"),
        ("5", "show_clients", "Clients"),
        ("6", "show_health", "Health"),
    ]

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Sidebar(id="sidebar")
            with Container(id="main-content"):
                with Horizontal(id="topbar"):
                    yield Static("", id="topbar-spacer")
                    with Vertical(id="topbar-center"):
                        yield Static("PINAK COMMAND CENTER", id="topbar-title")
                        yield Static("System Mesh • Global signals and memory substrate", id="topbar-subtitle")
                    yield Static("booting…", id="topbar-status")
                yield DashboardView(id="view-dashboard")
                yield AccessView(id="view-access")
                yield AgentsView(id="view-agents")
                yield ClientIssuesView(id="view-issues")
                yield ClientRegistryView(id="view-clients")
                yield HealthView(id="view-health")
        yield Footer()

    def on_mount(self):
        self.title = "Pinak Command Center"
        self.switch_tab("dashboard")
        self.set_interval(3, self.refresh_topbar_status)
        self.refresh_topbar_status()

    def action_show_dashboard(self):
        self.switch_tab("dashboard")

    def action_show_access(self):
        self.switch_tab("access")

    def action_show_agents(self):
        self.switch_tab("agents")

    def action_show_issues(self):
        self.switch_tab("issues")

    def action_show_clients(self):
        self.switch_tab("clients")

    def action_show_health(self):
        self.switch_tab("health")

    def on_button_pressed(self, event: Button.Pressed):
        button_id = event.button.id or ""
        if button_id.startswith("nav-"):
            tab = button_id.replace("nav-", "")
            if tab == "dashboard": self.switch_tab("dashboard")
            elif tab == "access": self.switch_tab("access")
            elif tab == "agents": self.switch_tab("agents")
            elif tab == "issues": self.switch_tab("issues")
            elif tab == "clients": self.switch_tab("clients")
            elif tab == "health": self.switch_tab("health")
            event.stop()

    def switch_tab(self, tab: str):
        self.query_one("#view-dashboard").display = (tab == "dashboard")
        self.query_one("#view-access").display = (tab == "access")
        self.query_one("#view-agents").display = (tab == "agents")
        self.query_one("#view-issues").display = (tab == "issues")
        self.query_one("#view-clients").display = (tab == "clients")
        self.query_one("#view-health").display = (tab == "health")
        title, subtitle = self.TAB_META.get(tab, ("Pinak Command Center", ""))
        self.query_one("#topbar-title", Static).update("PINAK COMMAND CENTER")
        self.query_one("#topbar-subtitle", Static).update(f"{title} • {subtitle}")
        self._set_nav_highlight(tab)

    def _set_nav_highlight(self, tab: str):
        nav_ids = [
            "nav-dashboard",
            "nav-access",
            "nav-agents",
            "nav-issues",
            "nav-clients",
            "nav-health",
        ]
        active_id = f"nav-{tab}"
        for nav_id in nav_ids:
            try:
                btn = self.query_one(f"#{nav_id}", Button)
            except Exception:
                continue
            if nav_id == active_id:
                btn.add_class("--highlight")
            else:
                btn.remove_class("--highlight")

    def refresh_topbar_status(self):
        db_path = "data/memory.db"
        status = "offline"
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                total = 0
                for t in ["memories_semantic", "memories_episodic", "memories_procedural", "memories_rag", "working_memory"]:
                    cur.execute(f"SELECT count(*) FROM {t}")
                    total += cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM logs_agents WHERE last_seen >= datetime('now','-60 seconds')")
                active_agents = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM logs_client_issues WHERE status = 'open'")
                open_issues = cur.fetchone()[0]
                conn.close()
                now_ist = datetime.datetime.now(IST).strftime("%H:%M IST")
                status = f"{now_ist} • m{total} • a{active_agents} • i{open_issues}"
            except Exception:
                status = "db error"
        self.query_one("#topbar-status", Static).update(status)

if __name__ == "__main__":
    MemoryApp().run()

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Header, Footer, Static, DataTable, Log, ListView, ListItem, Label, Button
from textual.reactive import reactive
from textual.screen import Screen
from textual.message import Message
import sqlite3
import os
import datetime
import faiss

# --- CSS ---
GLOBAL_CSS = """
Screen {
    background: #1e1e1e;
    color: #e0e0e0;
}

Header {
    background: #0d47a1;
    color: white;
    dock: top;
}

Footer {
    background: #0d47a1;
    color: white;
    dock: bottom;
}

/* Sidebar */
#sidebar {
    dock: left;
    width: 25;
    background: #252526;
    border-right: solid #333;
}

#sidebar Label {
    padding: 1;
    background: #333;
    color: #888;
    text-align: center;
    text-style: bold;
}

ListView {
    height: 100%;
}

ListItem {
    padding: 1 2;
    color: #cccccc;
}

ListItem:hover {
    background: #3e3e42;
}

ListItem.--highlight {
    background: #0d47a1;
    color: white;
    text-style: bold;
}

/* Main Content */
#main-content {
    padding: 1 2;
    height: 100%;
}

/* Dashboard Widgets */
.stat-card {
    background: #2d2d30;
    border: solid #3e3e42;
    height: 10;
    margin: 1;
    padding: 1;
    width: 1fr;
}

.stat-title {
    color: #4fc3f7;
    text-style: bold;
    border-bottom: solid #3e3e42;
}

.stat-value {
    text-align: center;
    color: #a5d6a7;
    text-style: bold;
    height: 1fr;
    content-align: center middle;
    font-size: 2; # Not supported in standard terminal but logic holds
}

/* Log */
Log {
    background: #1e1e1e;
    border: solid #3e3e42;
    height: 100%;
    color: #ce9178;
}

/* Data Table */
DataTable {
    background: #252526;
    border: solid #3e3e42;
}

.status-ok {
    color: #81c784;
}

.status-error {
    color: #e57373;
}
"""

class Sidebar(Container):
    def compose(self) -> ComposeResult:
        yield Label("PINAK MEMORY")
        yield ListView(
            ListItem(Label("📊  Dashboard"), id="nav-dashboard"),
            ListItem(Label("📡  Live Events"), id="nav-events"),
            ListItem(Label("👥  Agents"), id="nav-agents"),
            ListItem(Label("❤️   System Health"), id="nav-health"),
        )

class DashboardView(Container):
    def compose(self) -> ComposeResult:
        with Horizontal():
            yield Container(
                Static("Total Memories", classes="stat-title"),
                Static("Loading...", id="stat-total-mem", classes="stat-value"),
                classes="stat-card"
            )
            yield Container(
                Static("Active Tenants", classes="stat-title"),
                Static("Loading...", id="stat-tenants", classes="stat-value"),
                classes="stat-card"
            )
        with Horizontal():
             yield Container(
                Static("DB Size", classes="stat-title"),
                Static("Loading...", id="stat-db-size", classes="stat-value"),
                classes="stat-card"
            )
             yield Container(
                Static("Vector Index", classes="stat-title"),
                Static("Loading...", id="stat-vec-size", classes="stat-value"),
                classes="stat-card"
            )

    def on_mount(self):
        self.set_interval(2, self.refresh_stats)
        self.refresh_stats()

    def refresh_stats(self):
        db_path = "data/memory.db"
        vec_path = "data/vectors.index"

        total_mem = 0
        tenant_count = 0
        db_size = "0 MB"
        vec_size = "0"

        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                # Sum all layers
                count = 0
                for t in ["memories_semantic", "memories_episodic", "memories_procedural", "working_memory"]:
                    cur.execute(f"SELECT count(*) FROM {t}")
                    count += cur.fetchone()[0]
                total_mem = count

                cur.execute("SELECT count(DISTINCT tenant) FROM logs_session")
                tenant_count = cur.fetchone()[0]
                conn.close()

                size_mb = os.path.getsize(db_path) / (1024*1024)
                db_size = f"{size_mb:.2f} MB"
            except:
                pass

        if os.path.exists(vec_path):
            try:
                index = faiss.read_index(vec_path)
                vec_size = str(index.ntotal)
            except:
                vec_size = "Error"

        self.query_one("#stat-total-mem", Static).update(str(total_mem))
        self.query_one("#stat-tenants", Static).update(str(tenant_count))
        self.query_one("#stat-db-size", Static).update(db_size)
        self.query_one("#stat-vec-size", Static).update(vec_size)

class EventsView(Container):
    def compose(self) -> ComposeResult:
        yield Static("Live Audit Log (Tailing 'logs_events')", classes="stat-title")
        yield Log(id="event_log", highlight=True)

    def on_mount(self):
        self.last_ts = datetime.datetime.utcnow().isoformat()
        self.set_interval(1, self.tail)

    def tail(self):
        db_path = "data/memory.db"
        if not os.path.exists(db_path):
            return
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM logs_events WHERE ts > ? ORDER BY ts ASC", (self.last_ts,))
            rows = cur.fetchall()
            conn.close()

            log = self.query_one("#event_log", Log)
            for row in rows:
                ts = row['ts'].split('T')[1][:8]
                log.write_line(f"[{ts}] {row['tenant']}: {row['event_type']}")
                self.last_ts = row['ts']
        except:
            pass

class AgentsView(Container):
    def compose(self) -> ComposeResult:
        yield Static("Active Agents", classes="stat-title")
        yield DataTable(id="agents_table")

    def on_mount(self):
        table = self.query_one(DataTable)
        table.add_columns("Tenant", "Project", "Last Active", "Role")
        self.refresh_agents()
        self.set_interval(5, self.refresh_agents)

    def refresh_agents(self):
        table = self.query_one(DataTable)
        table.clear()

        db_path = "data/memory.db"
        if not os.path.exists(db_path):
            return

        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            # Get latest session for each tenant/project
            cur.execute("""
                SELECT tenant, project_id, max(ts), role
                FROM logs_session
                GROUP BY tenant, project_id
            """)
            rows = cur.fetchall()
            conn.close()

            for r in rows:
                table.add_row(r[0], r[1], r[2], r[3])
        except:
            pass

class HealthView(Container):
    def compose(self) -> ComposeResult:
        yield Static("System Integrity Check", classes="stat-title")
        yield VerticalScroll(Static("Running checks...", id="health_report"))
        yield Button("Run Doctor", id="btn_doctor", variant="primary")

    def on_button_pressed(self, event: Button.Pressed):
        if event.button.id == "btn_doctor":
            self.run_doctor()

    def on_mount(self):
        self.run_doctor()

    def run_doctor(self):
        report = ""
        # DB Check
        db_path = "data/memory.db"
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cur = conn.cursor()
                cur.execute("PRAGMA integrity_check")
                res = cur.fetchone()[0]
                conn.close()
                if res == "ok":
                    report += "[green]✅ SQLite Database Integrity: OK[/green]\n"
                else:
                    report += f"[red]❌ SQLite Integrity Error: {res}[/red]\n"
            except Exception as e:
                report += f"[red]❌ DB Check Failed: {e}[/red]\n"
        else:
            report += "[yellow]⚠️  Database file not found[/yellow]\n"

        # Vector Check
        vec_path = "data/vectors.index"
        if os.path.exists(vec_path):
            report += "[green]✅ Vector Index File: Present[/green]\n"
        else:
            report += "[yellow]⚠️  Vector Index not found[/yellow]\n"

        self.query_one("#health_report", Static).update(report)

class MemoryApp(App):
    CSS = GLOBAL_CSS
    BINDINGS = [("q", "quit", "Quit")]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal():
            yield Sidebar(id="sidebar")
            with Container(id="main-content"):
                yield DashboardView(id="view-dashboard")
                yield EventsView(id="view-events")
                yield AgentsView(id="view-agents")
                yield HealthView(id="view-health")
        yield Footer()

    def on_mount(self):
        self.title = "Pinak Memory Service"
        self.switch_tab("dashboard")

    def on_list_view_selected(self, event: ListView.Selected):
        nav_id = event.item.id
        if nav_id == "nav-dashboard":
            self.switch_tab("dashboard")
        elif nav_id == "nav-events":
            self.switch_tab("events")
        elif nav_id == "nav-agents":
            self.switch_tab("agents")
        elif nav_id == "nav-health":
            self.switch_tab("health")

    def switch_tab(self, tab: str):
        # Simple visibility toggle
        self.query_one("#view-dashboard").display = (tab == "dashboard")
        self.query_one("#view-events").display = (tab == "events")
        self.query_one("#view-agents").display = (tab == "agents")
        self.query_one("#view-health").display = (tab == "health")

if __name__ == "__main__":
    MemoryApp().run()

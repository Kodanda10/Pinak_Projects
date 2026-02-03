import typer
import uvicorn
import os
import sys
import json
import sqlite3
from typing import Optional
from pathlib import Path

# Add app to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import datetime
import jwt
import httpx
import subprocess
from app.core.database import DatabaseManager

app = typer.Typer(name="pinak-memory", help="Pinak Memory Service CLI")

def _get_db_path():
    # Load from config or default
    # Simplified: assume default for CLI
    return "data/memory.db"

def _get_vector_path():
    return "data/vectors.index"

@app.command()
def start(
    host: str = "0.0.0.0",
    port: int = 8001,
    reload: bool = False
):
    """Start the Pinak Memory Service."""
    typer.echo(f"Starting Memory Service on {host}:{port}...")
    uvicorn.run("app.main:app", host=host, port=port, reload=reload)

@app.command()
def doctor(fix: bool = False, heavy: bool = False):
    """Check system health and integrity."""
    from cli.doctor import run_doctor

    typer.echo("Running Pinak Memory Doctor...")
    report = run_doctor(fix=fix, allow_heavy=heavy)

    if report.actions:
        typer.echo("\n🛠️  Fixes Applied:")
        for action in report.actions:
            typer.echo(f" - {action}")

    if report.issues:
        typer.echo("\n⚠️  Issues Found:")
        for issue in report.issues:
            typer.echo(f" - {issue}")
    else:
        typer.echo("\n✅ All Systems Operational")

    if report.notes:
        typer.echo("\nℹ️  Notes:")
        for note in report.notes:
            typer.echo(f" - {note}")

@app.command()
def stats(json_output: bool = False):
    """Show usage statistics."""
    db_path = _get_db_path()
    if not os.path.exists(db_path):
        typer.echo("Database not found.")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    tables = ["memories_semantic", "memories_episodic", "memories_procedural", "memories_rag", "working_memory", "logs_events"]
    stats_data = {}

    for t in tables:
        try:
            cursor.execute(f"SELECT count(*) FROM {t}")
            stats_data[t] = cursor.fetchone()[0]
        except:
            stats_data[t] = 0

    # Storage Size
    stats_data['db_size_bytes'] = os.path.getsize(db_path)
    if os.path.exists(_get_vector_path()):
        stats_data['vector_size_bytes'] = os.path.getsize(_get_vector_path())

    conn.close()

    if json_output:
        typer.echo(json.dumps(stats_data, indent=2))
    else:
        typer.echo("\n📊 Memory Statistics")
        for k, v in stats_data.items():
            typer.echo(f"{k}: {v}")

@app.command()
def tui():
    """Launch Real-time Dashboard."""
    from cli.tui import MemoryApp
    MemoryApp().run()

@app.command("verify-mcp")
def verify_mcp():
    """Verify pinak-memory MCP config points to pinak_memory_mcp.py."""
    script = Path(__file__).resolve().parent.parent / "scripts" / "verify-mcp-config.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)
    if result.stdout:
        typer.echo(result.stdout.strip())
    if result.stderr:
        typer.echo(result.stderr.strip())

@app.command()
def mint(tenant: str, project: str = "default", secret: str = None):
    """Mint a development JWT token."""
    jwt_secret = secret or os.environ.get("PINAK_JWT_SECRET", "dev-secret-change-me")
    payload = {
        "sub": "local-dev",
        "tenant": tenant,
        "project_id": project,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=30),
    }
    token = jwt.encode(payload, jwt_secret, algorithm="HS256")
    typer.echo(token)

@app.command()
def search(query: str, tenant: str = "demo", project: str = "default", url: str = "http://localhost:8001"):
    """Perform a hybrid search (RRF) across all memory layers."""
    token_secret = os.environ.get("PINAK_JWT_SECRET", "dev-secret-change-me")
    token_payload = {
        "sub": "search-cli",
        "tenant": tenant,
        "project_id": project,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=5),
    }
    token = jwt.encode(token_payload, token_secret, algorithm="HS256")
    
    headers = {"Authorization": f"Bearer {token}"}
    try:
        with httpx.Client() as client:
            response = client.get(f"{url}/api/v1/memory/retrieve_context", params={"query": query}, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            typer.echo(f"\n🔍 Hybrid Search Results for: '{query}'")
            typer.echo("-" * 60)
            
            for layer, results in data.get("context_by_layer", {}).items():
                if results:
                    typer.echo(f"[{layer.upper()}]")
                    for r in results:
                        content = r.get("content") or r.get("steps") or str(r)
                        typer.echo(f" • {content}")
            
    except Exception as e:
        typer.echo(f"Error: {e}")

if __name__ == "__main__":
    app()

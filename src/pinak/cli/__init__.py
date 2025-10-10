from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Optional, TypeVar

import typer

from pinak.memory.manager import MemoryManager, MemoryManagerError


CONFIG_ENV_VAR = "PINAK_CONFIG_PATH"


app = typer.Typer(help="Pinak command-line interface for interacting with the Memory service.")
memory_app = typer.Typer(help="Commands for working with memories.")
events_app = typer.Typer(help="Commands for working with events.")
session_app = typer.Typer(help="Commands for working with sessions.")

app.add_typer(memory_app, name="memory")
app.add_typer(events_app, name="events")
app.add_typer(session_app, name="session")


@dataclass
class CLIConfig:
    """Runtime configuration persisted on disk for the CLI."""

    service_url: str = "http://localhost:8001"
    token: Optional[str] = None
    tenant_id: Optional[str] = None
    project_id: Optional[str] = None

    def to_dict(self) -> dict:
        data = asdict(self)
        # Persist only meaningful values
        return {key: value for key, value in data.items() if value is not None}


def get_config_path() -> Path:
    env_value = os.environ.get(CONFIG_ENV_VAR)
    if env_value:
        path = Path(env_value).expanduser()
        if path.is_dir():
            path = path / "config.json"
        return path

    return Path.home() / ".pinak" / "config.json"


def load_config() -> CLIConfig:
    path = get_config_path()
    if not path.exists():
        return CLIConfig()

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return CLIConfig()

    return CLIConfig(
        service_url=data.get("service_url", CLIConfig.service_url),
        token=data.get("token"),
        tenant_id=data.get("tenant_id"),
        project_id=data.get("project_id"),
    )


def save_config(config: CLIConfig) -> None:
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2))


def create_memory_manager(config: Optional[CLIConfig] = None) -> MemoryManager:
    config = config or load_config()
    return MemoryManager(
        service_base_url=config.service_url,
        token=config.token,
        tenant_id=config.tenant_id,
        project_id=config.project_id,
    )


_T = TypeVar("_T")


def _run_command(action: Callable[[], _T]) -> _T:
    """Execute a MemoryManager action and surface errors to the CLI."""

    try:
        return action()
    except MemoryManagerError as exc:
        typer.secho(str(exc), err=True, fg="red")
        raise typer.Exit(code=1) from exc


@app.command()
def login(
    service_url: Optional[str] = typer.Option(None, "--service-url", "-u", help="Base URL of the Pinak services."),
    token: Optional[str] = typer.Option(None, "--token", "-t", help="Authentication token for the service."),
    tenant_id: Optional[str] = typer.Option(None, "--tenant-id", help="Tenant identifier."),
    project_id: Optional[str] = typer.Option(None, "--project-id", help="Project identifier."),
) -> None:
    """Persist authentication and routing details for future CLI commands."""

    config = load_config()

    if service_url:
        config.service_url = service_url
    if token is not None:
        config.token = token
    if tenant_id is not None:
        config.tenant_id = tenant_id
    if project_id is not None:
        config.project_id = project_id

    if not config.token:
        raise typer.BadParameter("A token must be provided either now or previously via login.")

    manager = create_memory_manager(config)
    _run_command(lambda: manager.login(token=config.token, tenant_id=config.tenant_id, project_id=config.project_id))

    save_config(config)
    typer.echo("Login information saved.")


@memory_app.command("add")
def memory_add(
    content: str = typer.Argument(..., help="Content of the memory to store."),
    tags: Optional[List[str]] = typer.Option(
        None,
        "--tag",
        "-t",
        help="Tags to associate with the memory.",
        show_default=False,
    ),
) -> None:
    """Add a memory entry to the Pinak memory service."""

    manager = create_memory_manager()
    result = _run_command(lambda: manager.add_memory(content, list(tags) if tags else None))
    typer.echo(json.dumps(result, indent=2))


@memory_app.command("search")
def memory_search(
    query: str = typer.Argument(..., help="Search query."),
    k: int = typer.Option(5, "--k", help="Number of results to return."),
) -> None:
    """Search stored memories."""

    manager = create_memory_manager()
    results = _run_command(lambda: manager.search_memory(query, k=k))
    typer.echo(json.dumps(results, indent=2))


@events_app.command("list")
def events_list(
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Filter by query string."),
    since: Optional[str] = typer.Option(None, "--since", help="ISO timestamp for the start of the range."),
    until: Optional[str] = typer.Option(None, "--until", help="ISO timestamp for the end of the range."),
    limit: int = typer.Option(100, "--limit", help="Maximum number of events to fetch."),
    offset: int = typer.Option(0, "--offset", help="Pagination offset."),
) -> None:
    """List events stored in the memory service."""

    manager = create_memory_manager()
    events = _run_command(
        lambda: manager.list_events(query=query, since=since, until=until, limit=limit, offset=offset)
    )
    typer.echo(json.dumps(events, indent=2))


@session_app.command("list")
def session_list(
    session_id: str = typer.Argument(..., help="Identifier of the session to query."),
    limit: int = typer.Option(100, "--limit", help="Maximum number of session entries to fetch."),
    offset: int = typer.Option(0, "--offset", help="Pagination offset."),
    since: Optional[str] = typer.Option(None, "--since", help="ISO timestamp for the start of the range."),
    until: Optional[str] = typer.Option(None, "--until", help="ISO timestamp for the end of the range."),
) -> None:
    """List session entries from the memory service."""

    manager = create_memory_manager()
    items = _run_command(
        lambda: manager.list_session(session_id, limit=limit, offset=offset, since=since, until=until)
    )
    typer.echo(json.dumps(items, indent=2))


if __name__ == "__main__":
    app()

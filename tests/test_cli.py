from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import httpx
import pytest
from typer.testing import CliRunner

from pinak.cli import (
    CONFIG_ENV_VAR,
    CLIConfig,
    app,
    create_memory_manager,
    load_config,
    save_config,
)
from pinak.memory.manager import MemoryManager


runner = CliRunner()


@pytest.fixture(autouse=True)
def isolate_home(monkeypatch, tmp_path):
    """Ensure configuration is written into an isolated temporary directory."""

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    yield


def test_login_persists_configuration(tmp_path):
    result = runner.invoke(
        app,
        [
            "login",
            "--service-url",
            "http://example.com",
            "--token",
            "test-token",
            "--tenant-id",
            "tenant-123",
            "--project-id",
            "project-456",
        ],
    )

    assert result.exit_code == 0, result.stdout

    config_path = tmp_path / ".pinak" / "config.json"
    assert config_path.exists()

    stored = json.loads(config_path.read_text())
    assert stored["service_url"] == "http://example.com"
    assert stored["token"] == "test-token"
    assert stored["tenant_id"] == "tenant-123"
    assert stored["project_id"] == "project-456"


def test_config_path_can_be_overridden(tmp_path, monkeypatch):
    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    custom_path = custom_dir / "my-config.json"
    monkeypatch.setenv(CONFIG_ENV_VAR, str(custom_path))

    config = CLIConfig(service_url="http://override", token="secret")
    save_config(config)

    assert custom_path.exists()

    loaded = load_config()
    assert loaded.service_url == "http://override"
    assert loaded.token == "secret"


def test_memory_commands_use_memory_manager(monkeypatch):
    responses: Dict[str, httpx.Response] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/add"):
            payload = json.loads(request.content.decode())
            assert payload["content"] == "Remember this"
            assert payload["tags"] == ["cli", "test"]
            assert request.headers["Authorization"] == "Bearer token"
            return httpx.Response(200, json={"id": "1", "content": payload["content"]})
        if request.url.path.endswith("/search"):
            assert request.url.params["query"] == "remember"
            responses["search"] = httpx.Response(200, json=[{"content": "Remember this"}])
            return responses["search"]
        if request.url.path.endswith("/events"):
            responses["events"] = httpx.Response(200, json=[{"type": "info"}])
            return responses["events"]
        if request.url.path.endswith("/session/list"):
            assert request.url.params["session_id"] == "session-1"
            responses["session"] = httpx.Response(200, json=[{"session_id": "session-1", "content": "session data"}])
            return responses["session"]
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)

    def factory(config: CLIConfig) -> MemoryManager:
        client = httpx.Client(transport=transport)
        return MemoryManager(
            service_base_url=config.service_url,
            token=config.token,
            tenant_id=config.tenant_id,
            project_id=config.project_id,
            client=client,
        )

    # Persist configuration for the CLI to load
    config = CLIConfig(
        service_url="http://example.com",
        token="token",
        tenant_id="tenant",
        project_id="project",
    )
    save_config(config)

    monkeypatch.setattr("pinak.cli.create_memory_manager", lambda cfg=None: factory(cfg or load_config()))

    add_result = runner.invoke(app, ["memory", "add", "Remember this", "--tag", "cli", "--tag", "test"])
    assert add_result.exit_code == 0, add_result.stdout
    assert "Remember this" in add_result.stdout

    search_result = runner.invoke(app, ["memory", "search", "remember", "--k", "1"])
    assert search_result.exit_code == 0, search_result.stdout
    assert "Remember this" in search_result.stdout

    events_result = runner.invoke(app, ["events", "list", "--limit", "10"])
    assert events_result.exit_code == 0, events_result.stdout
    assert "info" in events_result.stdout

    session_result = runner.invoke(app, ["session", "list", "session-1"])
    assert session_result.exit_code == 0, session_result.stdout
    assert "session data" in session_result.stdout


def test_cli_surfaces_manager_errors(monkeypatch):
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "Internal error"})

    transport = httpx.MockTransport(handler)

    config = CLIConfig(service_url="http://example.com", token="token")
    save_config(config)

    def factory(cfg: CLIConfig) -> MemoryManager:
        client = httpx.Client(transport=transport)
        return MemoryManager(
            service_base_url=cfg.service_url,
            token=cfg.token,
            tenant_id=cfg.tenant_id,
            project_id=cfg.project_id,
            client=client,
        )

    monkeypatch.setattr("pinak.cli.create_memory_manager", lambda cfg=None: factory(cfg or load_config()))

    result = runner.invoke(app, ["memory", "search", "anything"])

    assert result.exit_code == 1
    assert "status 500" in result.output

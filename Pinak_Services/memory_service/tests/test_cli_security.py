import os
import pytest
from typer.testing import CliRunner
from cli.main import app

runner = CliRunner()

def test_mint_fails_without_secret(monkeypatch):
    """Test that mint command fails when PINAK_JWT_SECRET is unset."""
    if "PINAK_JWT_SECRET" in os.environ:
        monkeypatch.delenv("PINAK_JWT_SECRET")

    result = runner.invoke(app, ["mint", "test_tenant"])
    assert result.exit_code != 0
    # output contains both stdout and stderr by default in click/typer unless mixed=False
    # But let's check explicit property if available, or just output
    assert "PINAK_JWT_SECRET environment variable not set" in result.output

def test_mint_succeeds_with_env_secret(monkeypatch):
    """Test that mint command succeeds when PINAK_JWT_SECRET is set."""
    monkeypatch.setenv("PINAK_JWT_SECRET", "test-secret-123")

    result = runner.invoke(app, ["mint", "test_tenant"])
    assert result.exit_code == 0
    assert "eyJ" in result.stdout # JWT starts with eyJ

def test_mint_succeeds_with_arg_secret(monkeypatch):
    """Test that mint command succeeds when secret is passed as arg."""
    if "PINAK_JWT_SECRET" in os.environ:
        monkeypatch.delenv("PINAK_JWT_SECRET")

    result = runner.invoke(app, ["mint", "test_tenant", "--secret", "arg-secret"])
    assert result.exit_code == 0
    assert "eyJ" in result.stdout

def test_search_fails_without_secret(monkeypatch):
    """Test that search command fails when PINAK_JWT_SECRET is unset."""
    if "PINAK_JWT_SECRET" in os.environ:
        monkeypatch.delenv("PINAK_JWT_SECRET")

    result = runner.invoke(app, ["search", "test_query"])
    assert result.exit_code != 0
    assert "PINAK_JWT_SECRET environment variable not set" in result.output

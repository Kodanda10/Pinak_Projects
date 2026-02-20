
import jwt
import os
import pytest
from typer.testing import CliRunner
from cli.main import app

runner = CliRunner()

def test_mint_command_scopes(monkeypatch):
    # Set secret for test environment
    monkeypatch.setenv("PINAK_JWT_SECRET", "test-secret")

    # We invoke the command
    result = runner.invoke(app, ["mint", "test-tenant"])

    # Check exit code
    assert result.exit_code == 0, f"Command failed: {result.stdout}"

    # The output is the token
    token = result.stdout.strip()

    # Decode token
    # This might fail if the token is invalid or secret is wrong
    payload = jwt.decode(token, "test-secret", algorithms=["HS256"])

    # Verify scopes
    assert "scopes" in payload
    assert "memory.read" in payload["scopes"]
    assert "memory.write" in payload["scopes"]
    assert payload["tenant"] == "test-tenant"

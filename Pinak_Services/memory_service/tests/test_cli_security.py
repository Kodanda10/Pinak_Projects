import os
import pytest
from typer.testing import CliRunner
from cli.main import app
import jwt

runner = CliRunner()

def test_mint_without_secret_fails():
    # Ensure no env var is set
    if "PINAK_JWT_SECRET" in os.environ:
        del os.environ["PINAK_JWT_SECRET"]

    # Run mint command
    result = runner.invoke(app, ["mint", "test-tenant"])

    # Assert failure
    assert result.exit_code == 1
    assert "PINAK_JWT_SECRET not set" in result.stderr

def test_mint_with_secret_succeeds():
    os.environ["PINAK_JWT_SECRET"] = "secure-test-secret"

    result = runner.invoke(app, ["mint", "test-tenant"])

    assert result.exit_code == 0
    token = result.stdout.strip()

    # Verify token
    payload = jwt.decode(token, "secure-test-secret", algorithms=["HS256"])
    assert payload["tenant"] == "test-tenant"

def test_search_without_secret_fails():
    if "PINAK_JWT_SECRET" in os.environ:
        del os.environ["PINAK_JWT_SECRET"]

    result = runner.invoke(app, ["search", "test"])
    assert result.exit_code == 1
    assert "PINAK_JWT_SECRET not set" in result.stderr

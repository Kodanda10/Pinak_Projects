import json
import os
from pathlib import Path

from jose import jwt


def write_bridge_ctx(tmp: Path):
    pdir = tmp / ".pinak"
    pdir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "project_id": "Pnk-test-cli",
        "project_name": "test",
        "tenant": "default",
        "memory_url": "http://localhost:8011",
        "version": 1,
    }
    (pdir / "pinak.json").write_text(json.dumps(cfg), encoding="utf-8")


def test_pinak_token_generates_pid_and_role(tmp_path, monkeypatch, capsys):
    # Arrange: create a fake Bridge context
    write_bridge_ctx(tmp_path)
    monkeypatch.chdir(tmp_path)
    os.environ["SECRET_KEY"] = "unit-secret"

    # Act: call CLI main to mint a token with role
    from src.pinak.cli import main

    rc = main(
        ["token", "--sub", "alice", "--role", "editor", "--secret", "unit-secret"]
    )
    assert rc == 0
    token = capsys.readouterr().out.strip().splitlines()[-1]

    # Assert: token decodes and contains pid + role
    claims = jwt.decode(token, "unit-secret", algorithms=["HS256"])
    assert claims.get("pid") == "Pnk-test-cli"
    assert claims.get("role") == "editor"

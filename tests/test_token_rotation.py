import json
import os
from pathlib import Path

from jose import jwt


def write_bridge_ctx(tmp: Path):
    pdir = tmp / ".pinak"
    pdir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "project_id": "Pnk-rotate",
        "project_name": "test",
        "tenant": "default",
        "memory_url": "http://localhost:8011",
        "version": 1,
    }
    (pdir / "pinak.json").write_text(json.dumps(cfg), encoding="utf-8")


def test_token_rotation_sets_exp_and_stores(tmp_path, monkeypatch):
    write_bridge_ctx(tmp_path)
    monkeypatch.chdir(tmp_path)
    os.environ["SECRET_KEY"] = "unit-secret"
    from src.pinak.bridge.context import ProjectContext

    ctx = ProjectContext.find()
    tok = ctx.rotate_token(minutes=5, secret="unit-secret", sub="alice", role="editor")
    claims = jwt.decode(tok, "unit-secret", algorithms=["HS256"], options={"verify_exp": False})
    assert claims.get("pid") == "Pnk-rotate"
    assert claims.get("role") == "editor"
    assert int(claims.get("exp")) > 0
    # token retrievable via get_token
    assert ctx.get_token() == tok

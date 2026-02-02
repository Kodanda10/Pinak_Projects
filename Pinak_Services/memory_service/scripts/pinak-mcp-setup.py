#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any


AGENT_CONFIGS = {
    "codex": Path("~/.codex/mcp.json").expanduser(),
    "gemini": Path("~/.gemini/mcp.json").expanduser(),
    "cursor": Path("~/.cursor/mcp.json").expanduser(),
    "opencode": Path("~/.opencode/mcp_config.json").expanduser(),
    "kiro": Path("~/.kiro/settings/mcp.json").expanduser(),
    "antigravity": Path("~/.gemini/antigravity/mcp_config.json").expanduser(),
    "pi": Path("~/.pi/agent/settings.json").expanduser(),
}


def _ensure_pi_skill() -> None:
    skill_dir = Path("~/.pi/agent/skills/pinak-memory").expanduser()
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_path = skill_dir / "SKILL.md"
    content = """---
name: pinak-memory
description: Use pinak-memory when MCP tools are unavailable in Pi. Provides CLI commands for recall and remember.
alias: Pinak Memory
---

# Pinak Memory (Pi Skill Wrapper)

Pi does not natively support MCP tools. Use the pinak-memory CLI wrapper:

## Commands
- Status / summary:
  - ~/pinak-memory/bin/pinak-mcp status
- Recall:
  - ~/pinak-memory/bin/pinak-mcp recall "query"
- Remember episode:
  - ~/pinak-memory/bin/pinak-mcp remember-episode "goal" "outcome" "content" --tags tag1 tag2

## Notes
- The wrapper reads ~/pinak-memory/pinak.env for API URL and auth.
- If the wrapper is missing, run:
  - /path/to/memory_service/scripts/pinak-install-mcp.sh
"""
    skill_path.write_text(content)


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _ensure_mcp_server(cfg: Dict[str, Any], server_name: str, server_def: Dict[str, Any]) -> None:
    servers = cfg.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
    servers[server_name] = server_def
    cfg["mcpServers"] = servers


def main() -> int:
    parser = argparse.ArgumentParser(description="Pinak Memory MCP setup for common agents.")
    parser.add_argument("--agents", default="all", help="Comma-separated list (codex,gemini,pi,opencode,kiro,antigravity,cursor) or 'all'")
    parser.add_argument("--command", default=str(Path("~/pinak-memory/bin/pinak-mcp").expanduser()), help="Command for MCP server")
    parser.add_argument("--api-url", default=os.getenv("PINAK_API_URL", "http://127.0.0.1:8000/api/v1"))
    parser.add_argument("--token", default=os.getenv("PINAK_JWT_TOKEN", ""))
    parser.add_argument("--secret", default=os.getenv("PINAK_JWT_SECRET", ""))
    parser.add_argument("--client-id", default="")
    parser.add_argument("--client-name", default="")
    parser.add_argument("--project-id", default=os.getenv("PINAK_PROJECT_ID", "pinak-memory"))
    args = parser.parse_args()

    agent_list = [a.strip() for a in args.agents.split(",") if a.strip()]
    if args.agents == "all":
        agent_list = list(AGENT_CONFIGS.keys())

    repo_root = Path(__file__).resolve().parent.parent
    install_script = repo_root / "scripts" / "pinak-install-mcp.sh"
    if install_script.exists():
        os.system(f"bash \"{install_script}\"")

    command_path = args.command
    server_def = {
        "command": command_path,
        "args": [],
        "env": {},
    }
    if not Path(command_path).expanduser().exists():
        venv_py = repo_root / ".venv" / "bin" / "python"
        mcp_script = repo_root / "client" / "pinak_memory_mcp.py"
        if venv_py.exists() and mcp_script.exists():
            command_path = str(venv_py)
            server_def["command"] = command_path
            server_def["args"] = [str(mcp_script)]
    if "env" not in server_def:
        server_def["env"] = {}
    server_def["env"]["PINAK_API_URL"] = args.api_url
    server_def["env"]["PINAK_PROJECT_ID"] = args.project_id
    server_def["env"]["PINAK_AUTO_HEARTBEAT"] = "1"
    server_def["env"]["PINAK_STARTUP_BANNER"] = "1"
    if args.token:
        server_def["env"]["PINAK_JWT_TOKEN"] = args.token
    if args.secret and not args.token:
        server_def["env"]["PINAK_JWT_SECRET"] = args.secret
    env_path = Path("~/pinak-memory/pinak.env").expanduser()
    env_lines = [
        f"PINAK_API_URL={args.api_url}",
        f"PINAK_PROJECT_ID={args.project_id}",
        f"PINAK_HOME={repo_root}",
        f"PINAK_MCP_PYTHON={repo_root}/.venv/bin/python",
        "PINAK_AUTO_HEARTBEAT=1",
        "PINAK_STARTUP_BANNER=1",
    ]
    if args.token:
        env_lines.append(f"PINAK_JWT_TOKEN={args.token}")
    elif args.secret:
        env_lines.append(f"PINAK_JWT_SECRET={args.secret}")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(env_lines) + "\n")

    for agent in agent_list:
        path = AGENT_CONFIGS.get(agent)
        if not path:
            continue
        data = _load_json(path)
        per_def = json.loads(json.dumps(server_def))
        if args.client_id:
            per_def["env"]["PINAK_CLIENT_ID"] = args.client_id
        else:
            per_def["env"]["PINAK_CLIENT_ID"] = agent
        if args.client_name:
            per_def["env"]["PINAK_CLIENT_NAME"] = args.client_name
        else:
            per_def["env"]["PINAK_CLIENT_NAME"] = agent
        _ensure_mcp_server(data, "pinak-memory", per_def)
        _save_json(path, data)
        print(f"updated {path}")
        if agent == "pi":
            _ensure_pi_skill()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

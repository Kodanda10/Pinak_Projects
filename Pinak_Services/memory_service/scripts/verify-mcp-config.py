#!/usr/bin/env python3
import json
from pathlib import Path


CONFIG_PATHS = [
    Path("~/.codex/mcp.json").expanduser(),
    Path("~/.gemini/mcp.json").expanduser(),
    Path("~/.gemini/antigravity/mcp_config.json").expanduser(),
    Path("~/.cursor/mcp.json").expanduser(),
    Path("~/.opencode/mcp_config.json").expanduser(),
    Path("~/.kiro/settings/mcp.json").expanduser(),
    Path("~/.pi/agent/settings.json").expanduser(),
    Path("~/Library/Application Support/Claude/claude_desktop_config.json").expanduser(),
]


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    expected_cmd = str(repo_root / ".venv" / "bin" / "python")
    expected_script = str(repo_root / "client" / "pinak_memory_mcp.py")
    expected_args = [expected_script, "--mcp"]

    issues = 0
    checked = 0
    for path in CONFIG_PATHS:
        if not path.exists():
            continue
        data = _load(path)
        if not isinstance(data, dict):
            continue
        servers = data.get("mcpServers")
        if not isinstance(servers, dict) or "pinak-memory" not in servers:
            continue
        checked += 1
        server = servers["pinak-memory"] or {}
        cmd = server.get("command") or ""
        args = server.get("args") or []
        if cmd != expected_cmd or args != expected_args:
            issues += 1
            print(f"FAIL {path}")
            print(f"  command: {cmd}")
            print(f"  args:    {args}")
        else:
            print(f"OK   {path}")

    if checked == 0:
        print("No pinak-memory MCP configs found.")
        return 1
    if issues:
        print(f"{issues} config(s) do not point to pinak_memory_mcp.py.")
        return 2
    print("All pinak-memory MCP configs are correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

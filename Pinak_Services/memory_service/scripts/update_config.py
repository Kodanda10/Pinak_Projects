import json
import os
import sys

CONFIG_PATHS = [
    os.path.expanduser("~/Library/Application Support/Claude/claude_desktop_config.json"),
    os.path.expanduser("~/.gemini/mcp.json"),
    os.path.expanduser("~/.gemini/antigravity/mcp_config.json"),
    # New agents
    os.path.expanduser("~/.codex/mcp.json"),
    os.path.expanduser("~/.pi/agent/settings.json"),
    os.path.expanduser("~/.amp/mcp.json"),
    os.path.expanduser("~/.opencode/mcp_config.json"), 
    os.path.expanduser("~/.cursor/mcp.json")
]
PINAK_HOME = os.environ.get("PINAK_MCP_HOME", os.path.expanduser("~/pinak-memory"))
MCP_SCRIPT_PATH = os.path.expanduser(f"{PINAK_HOME}/bin/pinak-mcp")

def update_configs():
    # Define the Pinak Memory MCP config
    pinak_config = {
        "command": MCP_SCRIPT_PATH,
        "args": [],
        "env": {
            "PINAK_API_URL": os.environ.get("PINAK_API_URL", "http://127.0.0.1:8000/api/v1"),
            "PINAK_JWT_TOKEN": os.environ.get("PINAK_JWT_TOKEN", "SET_ME"),
            "PINAK_CLIENT_ID": os.environ.get("PINAK_CLIENT_ID", "SET_ME"),
            "PINAK_CLIENT_NAME": os.environ.get("PINAK_CLIENT_NAME", "SET_ME"),
            "PINAK_PROJECT_ID": os.environ.get("PINAK_PROJECT_ID", "pinak-memory"),
        },
    }

    for config_path in CONFIG_PATHS:
        print(f"Processing {config_path}...")
        
        if not os.path.exists(config_path):
            print(f"⚠️  Config file not found at {config_path}. Creating new.")
            config = {"mcpServers": {}}
            # Ensure directory exists
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
        else:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
            except json.JSONDecodeError:
                print(f"❌ Error decoding JSON at {config_path}. Skipping.")
                continue

        if "mcpServers" not in config:
            config["mcpServers"] = {}

        config["mcpServers"]["pinak-memory"] = pinak_config

        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Successfully updated 'pinak-memory' in {config_path}")
        except Exception as e:
            print(f"❌ Failed to write to {config_path}: {e}")

if __name__ == "__main__":
    update_configs()

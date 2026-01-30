import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

try:
    from client.pinak_memory_mcp import mcp
    print("✅ Successfully imported MCP Server")
    print(f"Server Name: {mcp.name}")
    print("\n📦 Registered Tools:")
    
    # Access the internal tool registry
    for tool_name, tool_func in mcp._tool_registry.items():
        print(f" - {tool_name}: {tool_func.__doc__.strip().splitlines()[0]}")
        
    print("\n✅ MCP Server Verification Complete")
except Exception as e:
    print(f"❌ Failed to inspect MCP: {e}")
    sys.exit(1)

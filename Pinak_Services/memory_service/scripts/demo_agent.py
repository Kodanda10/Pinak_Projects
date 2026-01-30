import sys
import os

# Ensure we can import the client package
sys.path.append(os.getcwd())

from client.pinak_memory_mcp import _recall_impl as recall

def run_agent_simulation():
    print("\n🤖 [AGENT] Simulation Started: 'Operations Agent 007'")
    print("---------------------------------------------------")
    
    query = "disk cleanup strategies"
    print(f"🤔 [AGENT] Thinking: I need to recall how to clean up disk space...")
    print(f"🔌 [MCP] Calling Tool: recall(query='{query}')")
    
    try:
        # Simulate the tool call
        memory_context = recall(query)
        
        print("\n✨ [MCP] Tool Output Received:")
        print("===================================================")
        print(memory_context)
        print("===================================================")
        
        print("\n✅ [AGENT] Simulation Complete. Memory integrated.")
        
    except Exception as e:
        print(f"❌ [AGENT] Error: {e}")

if __name__ == "__main__":
    run_agent_simulation()

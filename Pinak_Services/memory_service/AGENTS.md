# 🏹 Pinak Agent Protocols

Define instructions for specialist agents leveraging the Pinak substrate.

## Common Core Behaviors
- **Memory First**: Always check the memory service for previous context on a task to avoid duplication.
- **Trace Accountability**: Every significant discovery must be logged in the `Episodic` layer.
- **Continuous Learning**: Update the `Procedural` layer with new skill snippets or optimized commands.

## Agent Architecture
- **Tenant API**: `http://localhost:8000/api/v1/memory`
- **Default Tenant**: `default`
- **Default Project**: `pinak-memory`

## Observability
Use the **Agent Swarm** tab in the Pinak Command Center (`python cli/main.py tui`) to see who is active and what has been recorded.

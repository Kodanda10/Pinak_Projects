# Pinak One-Click Integration

## What is this?

This is a demonstration of **breeze-like integration** - adding memory capabilities to any Python project with just one command!

## Quick Start

```bash
# Run the integration script
python integrate_pinak.py
```

That's it! Your project now has memory capabilities.

## What happens during integration?

1. **Auto-detection**: Detects your project type (Django, Flask, FastAPI, etc.)
2. **Dependency Management**: Adds Pinak to your `requirements.txt`
3. **Configuration**: Creates a minimal `pinak_config.json` with sensible defaults
4. **Integration Code**: Generates `memory_integration.py` with ready-to-use examples
5. **Validation**: Tests that everything works correctly

## After Integration

Your project will have:

- ✅ `requirements.txt` updated with Pinak dependency
- ✅ `pinak_config.json` with configuration
- ✅ `memory_integration.py` with example usage
- ✅ Memory capabilities ready to use

## Example Usage

```python
from memory_integration import ProjectMemoryManager

# Initialize memory manager
memory = ProjectMemoryManager()

# Remember user actions
memory.remember_user_action(
    user_id="user123",
    action="created_project",
    context={"project_name": "my_app"}
)

# Store session context
memory.store_session_context(
    session_id="session_abc",
    context="User is working on authentication"
)

# Add knowledge to RAG system
memory.add_knowledge(
    topic="Authentication",
    content="JWT tokens should be validated on each request"
)

# Search knowledge base
results = memory.search_knowledge("authentication best practices")
```

## Configuration

Edit `pinak_config.json` to customize:

```json
{
  "memory_service_url": "http://localhost:8000",
  "api_key": "your_api_key_here",
  "default_project": "your_project_name",
  "auto_initialize": true,
  "enable_caching": true,
  "default_layers": ["episodic", "working", "session"]
}
```

## Starting the Memory Service

```bash
# Start the memory service
pinak-memory-service start

# Or use Docker
docker-compose up memory_service
```

## Features Added

- **Episodic Memory**: Remember user actions and events
- **Working Memory**: Temporary context for current tasks
- **Session Memory**: Context that persists during user sessions
- **RAG Integration**: Knowledge base with semantic search
- **Audit Trails**: Complete history of memory operations

## Zero-Config Experience

The integration script handles everything automatically:
- No manual configuration required
- Works with any Python project structure
- Compatible with major frameworks
- Minimal dependencies added
- Ready-to-use examples provided

This demonstrates the **breeze-like integration** that makes Pinak memory capabilities accessible to any developer with minimal effort!

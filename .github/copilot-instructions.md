# Pinak Project - Copilot Instructions

This document provides coding guidelines and context for GitHub Copilot when working on the Pinak project.

## Project Overview

Pinak is an enterprise-grade AI memory and context orchestrator with a local-first architecture. The project emphasizes security, multi-tenancy, and enterprise-ready features.

### Key Components

1. **Memory Service** (`Pinak_Services/memory_service/`): FastAPI-based service for 8-layer memory system
2. **CLI** (`src/pinak/cli/`): Command-line interface using Typer
3. **Security Auditor** (`src/pinak/security/`): Vulnerability scanning and compliance tools
4. **Memory Manager** (`src/pinak/memory/`): Client SDK for memory service interaction

## Architecture Principles

### 8-Layer Memory System

The project implements an enterprise-grade memory system with these layers:

1. **Semantic**: Vector embeddings for general knowledge (FAISS-based)
2. **Episodic**: Personal experiences with salience scoring
3. **Procedural**: Skills and step-by-step processes
4. **RAG**: Retrieval-augmented generation from external sources
5. **Events**: System and user events with timestamps
6. **Session**: Current session context with TTL
7. **Working**: Scratch/working memory with expiration
8. **Changelog**: Audit trail and redaction history

### Security-First Design

All code must follow these security principles:

- **JWT Authentication**: All API endpoints require Bearer tokens via `require_auth_context` dependency
- **Multi-tenancy**: Data is isolated by `tenant_id` and `project_id` at the filesystem level
- **Tamper-Evident Auditing**: Events use hash-chaining (`prev_hash` → `hash`) for integrity
- **Local-First**: Data stored locally by default with optional sync
- **No Hardcoded Secrets**: Use environment variables (e.g., `PINAK_JWT_SECRET`)

## Coding Standards

### Python Style

- **Version**: Python 3.11+ required
- **Type Hints**: Use type annotations for all function signatures
- **Async/Await**: Use async patterns for I/O operations in FastAPI endpoints
- **Dataclasses**: Prefer `@dataclass` for data structures (see `AuthContext`, `CLIConfig`)
- **Error Handling**: Raise descriptive exceptions; use FastAPI's `HTTPException` with proper status codes

### Testing Requirements (TDD)

- **Test-Driven Development**: Write tests before implementation
- **Location**: Tests in `tests/` or `Pinak_Services/memory_service/tests/`
- **Framework**: Use `pytest` with `pytest-asyncio` for async tests
- **Deterministic**: Tests use deterministic embeddings (no external model downloads)
- **Coverage**: Test authentication failures, tenant isolation, and audit verification

Example test pattern:
```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_requires_authentication(client: AsyncClient):
    response = await client.post("/api/v1/memory/add", json={...})
    assert response.status_code == 401  # Missing token
```

### API Development

#### FastAPI Patterns

- **Routing**: Use `APIRouter` in `app/api/v1/endpoints.py`
- **Dependencies**: Inject `AuthContext` via `Depends(require_auth_context)`
- **Schemas**: Define Pydantic models in `app/core/schemas.py`
- **Services**: Business logic in `app/services/memory_service.py`

Example endpoint structure:
```python
from fastapi import APIRouter, Depends
from app.core.security import require_auth_context, AuthContext

router = APIRouter()

@router.post("/add")
async def add_memory(
    content: str,
    auth: AuthContext = Depends(require_auth_context)
):
    # Use auth.tenant_id and auth.project_id for isolation
    ...
```

### Multi-Tenancy Implementation

Always scope data operations by tenant:

```python
# ✅ Correct: Tenant-scoped paths
data_dir = Path(f"data/{tenant_id}/{project_id}/semantic")

# ❌ Incorrect: Global data access
data_dir = Path("data/semantic")
```

### JWT Token Structure

Expected JWT claims:
```json
{
  "sub": "user-identifier",
  "tenant": "tenant-id",  // or "tenant_id"
  "project_id": "project-id",  // or "project"
  "roles": ["read", "write"],
  "iat": 1234567890,
  "exp": 1234567900
}
```

## Build and Development

### Setup Commands

```bash
# Install uv package manager
pip install --upgrade uv

# Sync dependencies
uv sync --extra tests --project Pinak_Services/memory_service --frozen

# Set JWT secret for development
export PINAK_JWT_SECRET="dev-secret-change-me"

# Run memory service
uv run --project Pinak_Services/memory_service uvicorn app.main:app --port 8001 --reload
```

### Running Tests

```bash
# Run all tests
cd Pinak_Services/memory_service
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=app --cov-report=html

# Run CI lint
flake8 Pinak_Services/memory_service --count --select=E9,F63,F7,F82 --show-source
```

### Docker

```bash
# Build and run
cd Pinak_Services/memory_service
docker build -t pinak-memory-service .
docker run -p 8001:8001 -e PINAK_JWT_SECRET="dev-secret" pinak-memory-service
```

## File Organization

### Directory Structure

```
Pinak_Projects/
├── .github/                    # CI/CD workflows and configs
│   ├── workflows/
│   └── copilot-instructions.md # This file
├── Pinak_Services/
│   └── memory_service/         # FastAPI memory service
│       ├── app/
│       │   ├── api/v1/         # API endpoints
│       │   ├── core/           # Security, schemas, config
│       │   ├── services/       # Business logic
│       │   └── main.py
│       ├── tests/              # Service-specific tests
│       ├── Dockerfile
│       └── pyproject.toml
├── src/pinak/                  # SDK and CLI
│   ├── cli/                    # Typer-based CLI
│   ├── memory/                 # MemoryManager client
│   └── security/               # SecurityAuditor
├── tests/                      # SDK/CLI tests
├── docs/                       # Documentation
├── config/                     # Security configurations
├── pyproject.toml              # Root project config
└── README.md
```

### Naming Conventions

- **Files**: Snake_case for Python modules (`memory_service.py`)
- **Classes**: PascalCase (`AuthContext`, `MemoryManager`)
- **Functions**: Snake_case (`require_auth_context`, `add_memory`)
- **Constants**: UPPER_SNAKE_CASE (`PINAK_JWT_SECRET`)

## Common Patterns

### CLI Commands (Typer)

```python
import typer
from pinak.memory.manager import MemoryManager

app = typer.Typer()

@app.command()
def add(content: str):
    """Add semantic memory."""
    config = load_config()
    manager = create_memory_manager(config)
    result = manager.add_semantic(content)
    typer.echo(f"Added: {result['id']}")
```

### Memory Manager Client

```python
from pinak.memory.manager import MemoryManager

manager = MemoryManager(
    service_base_url="http://localhost:8001",
    token="jwt-token-here",
    tenant_id="demo-tenant",
    project_id="demo-project"
)

# Add memory
result = manager.add_semantic("Python uses async/await", tags=["python"])

# Search
results = manager.search_semantic("async patterns", limit=5)
```

### Audit Hash Chaining

When implementing audit trails:

```python
import hashlib
import json

def _compute_audit_hash(event: dict, prev_hash: str) -> str:
    """Compute deterministic hash for event chaining."""
    payload = f"{prev_hash}|{json.dumps(event, sort_keys=True)}"
    return hashlib.sha256(payload.encode()).hexdigest()

# Usage in service
event = {
    "action": "memory_add",
    "ts": datetime.datetime.utcnow().isoformat(),
    ...
}
event["prev_hash"] = last_event_hash
event["hash"] = _compute_audit_hash(event, event["prev_hash"])
```

## Dependencies

### Core Dependencies

- **FastAPI** (>=0.111): Web framework for memory service
- **Uvicorn**: ASGI server
- **Sentence-Transformers** (>=2.2): Text embeddings
- **FAISS-CPU** (>=1.7.4): Vector search
- **PyJWT** (>=2.8): Token validation
- **Typer**: CLI framework
- **httpx**: HTTP client for SDK

### Testing Dependencies

- **pytest** (>=8.4.2): Test framework
- **pytest-asyncio** (>=1.2.0): Async test support
- **httpx**: Mocking HTTP requests
- **asgi-lifespan**: FastAPI test lifecycle

## Security Considerations

### What to Avoid

- ❌ Never commit JWT secrets or tokens
- ❌ Don't bypass `require_auth_context` dependency
- ❌ Avoid global data paths (always scope by tenant)
- ❌ Don't use `datetime.datetime.utcnow()` (deprecated; use `datetime.datetime.now(datetime.UTC)`)
- ❌ Don't make test-breaking changes without updating tests

### What to Do

- ✅ Always validate JWT tokens on API endpoints
- ✅ Use environment variables for secrets
- ✅ Implement tenant isolation at the storage layer
- ✅ Write tests for authentication failures
- ✅ Use hash-chaining for audit logs
- ✅ Follow TDD: tests first, then implementation

## Development Workflow

1. **Create Feature Branch**: `git checkout -b feature/your-feature`
2. **Write Tests First**: Add test cases in `tests/` or service-specific `tests/`
3. **Implement Feature**: Write minimal code to pass tests
4. **Run Tests**: `uv run pytest tests/ -v`
5. **Lint Code**: `flake8` for style checks
6. **Update Documentation**: Modify README or docs as needed
7. **Submit PR**: Include test results and documentation updates

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci-cd.yml`) runs on every push:

1. **Test**: Runs pytest with coverage
2. **Lint**: Flake8 static analysis
3. **Build**: Docker image build
4. **Security Scan**: Trivy vulnerability scanning
5. **Audit**: pip-audit for dependency vulnerabilities

Ensure all checks pass before merging.

## Documentation Standards

- **Docstrings**: Use descriptive docstrings for public functions/classes
- **README**: Keep README.md updated with new features
- **API Docs**: FastAPI auto-generates OpenAPI docs at `/docs`
- **Inline Comments**: Minimal comments; prefer self-documenting code

## Version and License

- **Version**: 0.1.0-alpha (Phase 1: Core Intelligence)
- **Python**: >=3.11
- **License**: MIT

## Resources

- **Repository**: https://github.com/Pinak-Setu/Pinak_Projects
- **Documentation**: See `docs/` directory
- **Remediation Plan**: `docs/remediation_plan.md` for security hardening details
- **Issues**: GitHub Issues for bug reports and feature requests

## Special Notes for Copilot

When generating code for this project:

1. **Prioritize Security**: Always include authentication checks
2. **Respect Multi-Tenancy**: Never mix data between tenants
3. **Follow TDD**: Generate tests alongside implementation
4. **Use Type Hints**: All functions should have type annotations
5. **Async Patterns**: Use `async`/`await` for FastAPI endpoints
6. **Deterministic Tests**: Avoid external API calls or model downloads in tests
7. **Environment Variables**: Never hardcode secrets
8. **Error Messages**: Provide clear, actionable error messages

---

*Last updated: 2025-10-11*

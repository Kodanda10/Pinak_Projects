# GitHub Copilot Instructions for Pinak

This repository contains Pinak, an enterprise-grade AI Memory & Context Orchestrator with local-first architecture, JWT-based authentication, and multi-tenant isolation.

## Project Overview

- **Language**: Python 3.11+
- **Framework**: FastAPI for memory service, Typer for CLI
- **Architecture**: Multi-tenant, local-first with JWT authentication
- **Testing**: pytest with async support, TDD approach
- **Package Manager**: uv (fast Python package manager)

## Development Philosophy

1. **Test-Driven Development (TDD)**: Always write tests first before implementing functionality
2. **Security First**: All endpoints require JWT authentication with tenant/project isolation
3. **Local-First**: Data stored locally by default with file-system level tenant segregation
4. **Enterprise-Ready**: Focus on auditability, compliance, and tamper-evident logs

## Code Style and Conventions

### Python Style

- Use Python 3.11+ features including type hints from `typing` module
- Follow PEP 8 conventions
- Use `from __future__ import annotations` at the top of files with type hints
- Prefer `Optional[Type]` over `Type | None` for clarity
- Use dataclasses for configuration and data structures
- Organize imports: standard library, third-party, local modules (separated by blank lines)

### Documentation

- Use docstrings for all public functions and classes
- Follow Google-style docstrings format
- Document security-relevant behavior explicitly
- Include type hints in function signatures

### Naming Conventions

- Use snake_case for functions, variables, and methods
- Use PascalCase for classes
- Use UPPER_SNAKE_CASE for constants
- Prefix private methods/functions with single underscore `_`
- Use descriptive names: `tenant_id`, `project_id`, `memory_service`, not abbreviations

## Security Practices

### Authentication and Authorization

- **Always require JWT authentication** for API endpoints
- JWT tokens must include `tenant` (or `tenant_id`) and `project_id` claims
- Use `require_auth_context` dependency for FastAPI routes
- Never log or expose JWT secrets or tokens in responses
- Use Bearer token format: `Authorization: Bearer <token>`

### Multi-Tenancy

- All data storage must be tenant-scoped using sanitized directory structures
- Use `_sanitize_component()` to clean tenant/project identifiers (strip special chars)
- Store data at: `{data_root}/{tenant_component}/{project_component}/`
- Never allow cross-tenant data access
- Validate tenant/project from JWT claims, not from request parameters

### Audit and Compliance

- Implement tamper-evident audit logs using hash chaining
- Use `_compute_audit_hash()` for deterministic hashing
- Store `prev_hash` and `hash` in every audit entry
- Append audit entries to JSONL files with `_append_audit_jsonl()`
- Never modify or delete audit logs programmatically

### Data Handling

- Store sensitive data locally by default
- Use JSONL format for layer persistence (episodic, procedural, RAG, events)
- Never include passwords, secrets, or sensitive tokens in code or logs
- Implement configurable redaction rules for privacy

## Testing Standards

### Test Structure

- Use pytest with pytest-asyncio for async tests
- Place tests in `tests/` directory within each service/package
- Use fixtures for dependency injection and test setup
- Follow naming: `test_<functionality>.py` for test files, `test_<feature>_<scenario>` for test functions

### Test Configuration

- Set environment variables in fixtures: `PINAK_JWT_SECRET`, `PINAK_EMBEDDING_BACKEND`
- Use `tmp_path` fixture for isolated file system operations
- Use deterministic embeddings (`PINAK_EMBEDDING_BACKEND=dummy`) to avoid model downloads
- Override dependencies with `app.dependency_overrides` for FastAPI testing

### Test Coverage Requirements

- Write unit tests for all business logic
- Test authentication rejection paths (missing/invalid tokens)
- Test multi-tenant isolation (ensure data segregation)
- Test audit chain integrity (hash verification)
- Achieve minimum 90% code coverage for new features
- Use `pytest --cov=app --cov-report=html` for coverage reports

### Example Test Pattern

```python
@pytest.mark.asyncio
async def test_feature_requires_authentication(test_app):
    async with LifespanManager(test_app):
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url="http://test") as client:
            response = await client.get("/api/v1/endpoint")
    assert response.status_code == 401
```

## API Development

### FastAPI Routes

- Use versioned API paths: `/api/v1/memory/...`
- Apply `require_auth_context` dependency to all routes
- Extract tenant/project from auth context, not request body
- Return appropriate HTTP status codes (401 for auth, 403 for authorization, 404 for not found)
- Use Pydantic schemas for request/response validation

### Memory Service Patterns

- Initialize with config from `config.json` or environment variables
- Support both real and deterministic embeddings (for testing)
- Lazy-load FAISS indexes per tenant/project combination
- Store metadata alongside vectors in JSONL files
- Use date-based file naming: `{layer}_{YYYYMMDD}.jsonl`

## CLI Development

### Typer CLI Structure

- Use Typer for CLI framework
- Organize commands into sub-apps (memory_app, events_app, session_app)
- Persist configuration in `~/.pinak/config.json`
- Support environment variable `PINAK_CONFIG_PATH` for custom config location
- Provide helpful error messages using `typer.secho(err=True, fg="red")`

### Configuration Management

- Store: `service_url`, `token`, `tenant_id`, `project_id` in config
- Load config with fallbacks to defaults
- Validate required fields (token is mandatory for authenticated operations)
- Use `MemoryManager` client for API interactions

## Dependencies and Package Management

### Using uv

- Use `uv sync --frozen` for reproducible installs
- Use `uv run` for executing commands in project environment
- For memory service: `uv sync --project Pinak_Services/memory_service`
- For tests: `uv sync --project Pinak_Services/memory_service --group tests --frozen`

### Adding Dependencies

- Add to `pyproject.toml` under `[project.dependencies]`
- Use optional dependencies for test/dev tools: `[project.optional-dependencies]`
- Pin major versions but allow minor updates: `fastapi>=0.111`
- Run `uv sync` after adding dependencies

### Workspace Structure

- Root project defines CLI and shared libraries
- Memory service is a workspace member in `Pinak_Services/memory_service`
- Each service has its own `pyproject.toml`

## Architecture Patterns

### 8-Layer Memory System

1. **Semantic**: General knowledge vector embeddings
2. **Episodic**: Personal experiences with salience scores
3. **Procedural**: Step-by-step processes and skills
4. **RAG**: Retrieval-augmented generation from external sources
5. **Events**: System/user events with timestamps
6. **Session**: Current session context with TTL
7. **Working**: Scratch memory with expiration
8. **Changelog**: Audit trail and redaction history

### Storage Patterns

- FAISS for vector similarity search
- JSONL for layer persistence (append-only)
- Redis for optional caching (must be optional, not required)
- File-system directories for tenant/project isolation

### Service Architecture

- Memory Service: Core vector storage/retrieval (FastAPI)
- CLI: Command-line interface (Typer)
- MemoryManager: Python client library for API interaction

## Common Pitfalls to Avoid

1. **Don't bypass authentication**: Never create unauthenticated routes
2. **Don't mix tenant data**: Always scope operations to tenant/project from JWT
3. **Don't hardcode credentials**: Use environment variables for secrets
4. **Don't skip tests**: TDD is mandatory, write tests first
5. **Don't use global state**: Inject dependencies properly
6. **Don't ignore audit logs**: Always log security-relevant events
7. **Don't download models in tests**: Use `PINAK_EMBEDDING_BACKEND=dummy`
8. **Don't modify working code unnecessarily**: Make minimal, surgical changes

## Environment Variables

- `PINAK_JWT_SECRET`: Secret for JWT signing/verification (required for production)
- `PINAK_EMBEDDING_BACKEND`: Set to "dummy" for tests to use deterministic embeddings
- `PINAK_CONFIG_PATH`: Custom path for CLI configuration file
- `PINAK_DEV_TOKEN`: Development JWT token for manual testing

## Running Commands

### Development Server

```bash
export PINAK_JWT_SECRET="dev-secret-change-me"
uv run --project Pinak_Services/memory_service uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### Testing

```bash
# Run all tests with deterministic embeddings
uv sync --project Pinak_Services/memory_service --group tests --frozen
uv run --project Pinak_Services/memory_service pytest tests/ -v

# Run with coverage
uv run --project Pinak_Services/memory_service pytest tests/ --cov=app --cov-report=html
```

### CLI Usage

```bash
# Configure authentication
uv run pinak login --token <jwt-token> --tenant-id dev --project-id test

# Add memory
uv run pinak memory add "Important information" --tags feature,api

# Search memories
uv run pinak memory search "important"
```

## Documentation Standards

- Update README.md for user-facing changes
- Update docs/remediation_plan.md for security/architecture changes
- Document all public APIs with request/response examples
- Include curl examples with proper JWT authentication
- Keep CONTRIBUTING.md updated with development practices

## Git Workflow

- Create feature branches from main
- Follow conventional commits (optional but encouraged)
- Ensure all tests pass before committing
- Update documentation in the same commit as code changes
- Use `.gitignore` to exclude: `__pycache__/`, `*.pyc`, `.venv/`, `data/`, `*.egg-info/`

## Enterprise Features Focus

- **Tamper-Evident Logs**: Hash-chained audit entries
- **Tenant Isolation**: File-system level segregation
- **JWT Authentication**: All endpoints protected
- **Privacy**: Configurable redaction rules
- **Compliance**: GDPR, SOC2 ready architecture
- **Observability**: Structured logging, metrics ready

## Future Considerations

- Pinakontext SOTA orchestrator (Phase 2)
- Hybrid retrieval (BM25 + semantic)
- OPA/Rego policy engine integration
- OTEL observability and Prometheus metrics
- Federated learning for privacy
- Production deployment guides

# 🏹 Pinak — Enterprise-Grade AI Memory & Context Orchestrator

**Version:** 0.1.0-alpha  
**Status:** Active Development (Phase 1: Core Intelligence)  
**License:** MIT  
**Repository:** https://github.com/Pinak-Setu/Pinak_Projects

## 📋 Overview

Pinak is a local-first, enterprise-grade AI assistant designed for developers, providing unparalleled AI memory, security auditing, CLI interaction logging, and real-time context orchestration. Built with state-of-the-art security baselines and offline functionality.

### Key Features
- **8-Layer Memory System**: Semantic, Episodic, Procedural, RAG, Events, Session, Working, Changelog
- **Local-First Architecture**: Data stored locally with optional sync
- **Enterprise Security Foundations**: JWT-protected APIs and tamper-evident audits
- **Real-Time Context**: Proactive nudges and orchestration
- **Multi-Tenant**: Project-based isolation with audit trails

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (optional)
- Redis (optional, for caching)
- [uv](https://astral.sh/uv) (fast Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/Pinak-Setu/Pinak_Projects.git
cd Pinak_Projects

# Install uv (skip if already installed)
pip install --upgrade uv

# Sync project dependencies
uv sync --frozen
uv sync --project Pinak_Services/memory_service --no-dev --frozen

# Configure authentication secret (generate a strong value in production)
export PINAK_JWT_SECRET="dev-secret-change-me"

# Run the service
uv run --project Pinak_Services/memory_service uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### Authentication & Multi-Tenancy

All API routes require a Bearer JWT. The token **must** include `tenant` (or `tenant_id`) and `project_id` claims; data is stored and retrieved from segregated directories derived from these values.

In development you can mint a token with:

```bash
python - <<'PY'
import datetime, jwt
token = jwt.encode(
    {
        "sub": "local-dev",
        "tenant": "demo-tenant",
        "project_id": "demo-project",
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1),
    },
    "dev-secret-change-me",
    algorithm="HS256",
)
print(token)
PY
```

Then export it for subsequent requests:

```bash
export PINAK_DEV_TOKEN="<token-from-command-above>"
```

Use the generated value in the `Authorization: Bearer <token>` header when calling the API.

### API Usage

```bash
# Add semantic memory
curl -X POST "http://localhost:8001/api/v1/memory/add" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${PINAK_DEV_TOKEN}" \
  -d '{"content": "Python async functions use await", "tags": ["python", "async"]}'

# Search memory
curl "http://localhost:8001/api/v1/memory/search?query=async&limit=5" \
  -H "Authorization: Bearer ${PINAK_DEV_TOKEN}"

# Add episodic memory
curl -X POST "http://localhost:8001/api/v1/memory/episodic/add" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${PINAK_DEV_TOKEN}" \
  -d '{"content": "Fixed async bug in CI pipeline", "salience": 8}'

# Multi-layer search
curl "http://localhost:8001/api/v1/memory/search_v2?query=async&layers=episodic,procedural,rag" \
  -H "Authorization: Bearer ${PINAK_DEV_TOKEN}"
```

## 🏗️ Architecture

### Memory Layers
1. **Semantic**: Vector embeddings for general knowledge
2. **Episodic**: Personal experiences with salience scoring
3. **Procedural**: Skills and step-by-step processes
4. **RAG**: Retrieval-augmented generation from external sources
5. **Events**: System and user events with timestamps
6. **Session**: Current session context with TTL
7. **Working**: Scratch/working memory with expiration
8. **Changelog**: Audit trail and redaction history

### Services
- **Memory Service**: Core vector storage and retrieval (FastAPI)
- **Governance Gateway**: Authentication and policy enforcement
- **Security Auditor**: Vulnerability scanning and compliance
- **CLI Logger**: Command auditing and privacy redaction

## 📊 Development Roadmap

### Phase 1: Core Intelligence (Current)
- ✅ 8-layer memory system implementation
- ✅ Basic vector search with FAISS
- ✅ JSONL storage for layers
- 🔄 Unit tests and TDD
- 🔄 CI/CD pipeline
- 🔄 Documentation

### Phase 2: Advanced Intelligence
- Pinakontext SOTA orchestrator
- Hybrid retrieval (BM25 + semantic)
- Recipe engine for context synthesis
- OTEL observability and Prometheus metrics

### Phase 3: Enterprise Readiness
- Multi-tenant database integration
- OPA/Rego policy engine
- JWT/OIDC authentication
- Audit chain verification

### Phase 4: Ecosystem & Scale
- macOS app with PyInstaller
- API marketplace
- Federated learning for privacy
- Production deployment guides

## 🧪 Testing

```bash
# Run all tests (uses deterministic embeddings, no external downloads)
uv sync --project Pinak_Services/memory_service --group tests --frozen
uv run --project Pinak_Services/memory_service pytest tests/ -v

# Run with coverage
uv run --project Pinak_Services/memory_service pytest tests/ --cov=app --cov-report=html

# Run demo script
uv run --project Pinak_Services/memory_service python ../../scripts/demo_all_layers.py
```

## 🔒 Security

- **Local-First**: All data stored locally by default
- **JWT Guarded Endpoints**: All memory APIs enforce Bearer token validation
- **Tamper-Evident**: Hash-chained audit logs for events
- **Tenant Isolation**: File-system level segregation by `tenant` and `project_id`
- **Privacy**: Configurable redaction rules
- **Compliance**: GDPR, SOC2, and enterprise-ready

See [SECURITY.md](SECURITY.md) for details.

## 📚 Documentation

- [Enterprise Reference Architecture](pinak_enterprise_reference.md)
- [SOTA Context Orchestrator Plan](Pinakontext_SOTA_Plan.md)
- [Development Log](development_log.md)
- [API Documentation](docs/)
- [Remediation & Hardening Plan](docs/remediation_plan.md)
- [Contributing Guide](CONTRIBUTING.md)

## 🤝 Contributing

We follow TDD, strict file management, and enterprise security protocols.

1. Fork the repository
2. Create a feature branch
3. Write tests first
4. Implement functionality
5. Ensure CI passes
6. Submit PR with comprehensive docs

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/Pinak-Setu/Pinak_Projects/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Pinak-Setu/Pinak_Projects/discussions)
- **Email**: support@pinak-setu.com

---

**Built with ❤️ for developers, by developers.**  
*Last updated: 2025-08-29*

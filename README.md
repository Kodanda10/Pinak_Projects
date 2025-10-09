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
- **Enterprise Security**: JWT, OPA/Rego policies, tamper-evident audits
- **Real-Time Context**: Proactive nudges and orchestration
- **Multi-Tenant**: Project-based isolation with audit trails

## 🚀 Quick Start

### Prerequisites
- Python 3.13+
- Docker (optional)
- Redis (optional, for caching)

### Installation

```bash
# Clone the repository
git clone https://github.com/Pinak-Setu/Pinak_Projects.git
cd Pinak_Projects

# Install Python dependencies for the CLI and tests
pip install -r requirements.txt

# (Optional) Run the FastAPI memory service if you want to talk to a live backend
cd Pinak_Services/memory_service
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### API Usage

```bash
# Add semantic memory
curl -X POST "http://localhost:8001/api/v1/memory/add" \
  -H "Content-Type: application/json" \
  -d '{"content": "Python async functions use await", "tags": ["python", "async"]}'

# Search memory
curl "http://localhost:8001/api/v1/memory/search?query=async&limit=5"

# Add episodic memory
curl -X POST "http://localhost:8001/api/v1/memory/episodic/add" \
  -H "Content-Type: application/json" \
  -d '{"content": "Fixed async bug in CI pipeline", "salience": 8}'

# Multi-layer search
curl "http://localhost:8001/api/v1/memory/search_v2?query=async&layers=semantic,episodic,procedural"
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
# Run the CLI integration suite
PYTHONPATH=src pytest tests/test_cli.py -q

# Run the entire test suite
PYTHONPATH=src pytest

# Legacy service tests are still available under the FastAPI app
cd Pinak_Services/memory_service
pytest tests/ -v
```

### CLI Quick Start

The Typer-based `pinak` CLI persists its configuration in `~/.pinak/config.json` by default.

```bash
# Log in and persist configuration
pinak login --service-url http://localhost:8001 --token <TOKEN> --tenant-id <TENANT> --project-id <PROJECT>

# Add a memory and search for it
pinak memory add "Remember this" --tag cli --tag example
pinak memory search "Remember" --k 3

# List events and session data
pinak events list --limit 10
pinak session list SESSION-ID --limit 10

# Override the configuration location (optional)
PINAK_CONFIG_PATH=/tmp/pinak-config.json pinak memory search "query"
```

## 🔒 Security

- **Local-First**: All data stored locally by default
- **Tamper-Evident**: Hash-chained audit logs
- **Privacy**: Configurable redaction rules
- **Compliance**: GDPR, SOC2, and enterprise-ready

See [SECURITY.md](SECURITY.md) for details.

## 📚 Documentation

- [Enterprise Reference Architecture](pinak_enterprise_reference.md)
- [SOTA Context Orchestrator Plan](Pinakontext_SOTA_Plan.md)
- [Development Log](development_log.md)
- [API Documentation](docs/)
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

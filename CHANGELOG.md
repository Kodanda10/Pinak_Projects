## 1.3.0 (2025-08-29) - MAJOR RELEASE: Complete 8-Layer Memory System ✅

### 🎉 **MAJOR ACHIEVEMENT: Production-Ready Memory Service**

**8-Layer Memory Architecture:**
- **Semantic Layer**: FAISS-based vector similarity search with SentenceTransformers embeddings
- **Episodic Layer**: JSONL-based storage with salience scoring for important memories
- **Procedural Layer**: Skill-based memory storage with step-by-step instructions
- **RAG Layer**: External knowledge integration with source attribution
- **Events Layer**: Audit trail logging with timestamp filtering and search
- **Session Layer**: TTL-based temporary memory with automatic expiration
- **Working Layer**: Scratch memory for immediate context and temporary data
- **Changelog Layer**: Redaction and versioning capabilities with tamper-evident logging

**FastAPI Service Implementation:**
- **17 REST Endpoints**: Complete API coverage for all memory layers
- **Async Support**: High-performance async operations throughout
- **Multi-tenant Architecture**: Project-based isolation with proper data segregation
- **Cross-Layer Search**: Unified search across all memory layers simultaneously
- **Interactive Documentation**: Auto-generated API docs at `/docs`

**Enterprise-Grade Features:**
- **Security**: JWT authentication, audit trails, and secure data handling
- **Persistence**: JSONL-based storage with timestamp tracking and integrity
- **Caching**: Optional Redis integration with graceful fallback
- **Type Safety**: Full Pydantic validation and comprehensive error handling
- **Testing**: 17/17 tests passing with comprehensive coverage

**CI/CD & DevOps:**
- **GitHub Actions**: Automated testing, building, and deployment pipeline
- **Docker Support**: Containerization ready for production deployment
- **Documentation**: Updated README, API docs, and comprehensive guides

**Quality Assurance:**
- **TDD Compliance**: All development followed strict Test-Driven Development
- **Test Coverage**: 17 passing tests covering unit, integration, and async scenarios
- **Code Quality**: Enterprise-grade architecture with clean separation of concerns

### 🔄 **Migration Notes**
- Memory service is backward compatible with existing implementations
- Redis is optional - service works without Redis for local development
- All data is stored locally ensuring privacy and offline accessibility

## 1.2.0 (2025-08-28) - Strategic Roadmap & New Initiatives

- **Vision Alignment**: Initiating a comprehensive roadmap to evolve Pinak into a world-class, SOTA, enterprise-grade macOS application (Pinak Bridge), adhering to FANG/Apple levels of excellence.
- **New Core Component - Pinakontext**: Introduction of Pinakontext, a SOTA Context Orchestrator, designed for real-time, on-demand, and nudge-based context for developers and AI agents. This includes: 
    - Initial scaffolding for `pinak-context` CLI.
    - Core components: `ContextBroker`, `NudgeManager`, `Policy Gate` (initial stubs).
    - Integration of `pyyaml`, `opentelemetry-api`, `prometheus-client`, `portalocker` dependencies.
- **New Feature - CLI Interaction Logging**: Beginning implementation of comprehensive CLI interaction logging for enhanced developer context and auditability. This includes: 
    - Finalization of event schema and redaction rules.
    - Development of `pinak-clips` (local capture agent).
    - Initial Zsh shell hooks and basic CLI commands (`pinak session log`).
- **Enhanced Memory Service**: Commencing foundational work for advanced vector indexing, persistent metadata storage, and deeper Redis integration.
- **Enhanced Security Auditor**: Starting implementation of recursive secret scanning, entropy-based detection, and multi-language dependency scanning.
- **Development Protocols**: Formalizing strict adherence to Test-Driven Development (TDD), CI/CD pipelines, continuous GitHub repository updates, and automated documentation. 
- **File Management Policy**: Implementing a strict 'no file deletion' policy; unused files will be moved to a local archive with a reason.

## 1.1.1 (2025-08-28)

- Fix: Correct JWT `exp` calculation using timezone-aware UTC to prevent immediate expiry (401 issues resolved).
- Bridge: `pinak-bridge verify` now reports token expiry and seconds to expiry.
- UX: Memory client prints guidance on 401 to rotate token.
- New: macOS menu bar app `pinak-menubar` with status + self-healing.


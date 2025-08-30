# Pinak - The Local-First AI Assistant for Developers

![Pinak Sync Image](pinak-sync.png)

Pinak is evolving into a cutting-edge, ultra-modern, and highly efficient local-first AI assistant, designed to empower developers with unparalleled capabilities in AI memory, security auditing, CLI interaction logging, and real-time context orchestration. Our vision is to build a world-class macOS application that sets new standards for developer tools, maintaining excellence akin to leading technology companies.

## Core Capabilities

### 🧠 AI Memory Service ✅ **COMPLETED**

Pinak provides a robust AI memory service that allows developers to store, manage, and semantically search through their knowledge. This service is designed for efficient retrieval of information, enabling developers to quickly access relevant code snippets, documentation, decisions, and insights.

**✅ IMPLEMENTED FEATURES:**
*   **8 Memory Layers:** Semantic, Episodic, Procedural, RAG, Events, Session, Working, Changelog
*   **Semantic Search:** Leverage advanced vector embeddings and similarity search for intelligent information retrieval.
*   **Local-First Storage:** Data is stored locally, ensuring privacy, speed, and offline accessibility.
*   **Cross-Layer Search:** Search across all memory layers simultaneously
*   **FastAPI REST API:** 17 endpoints with comprehensive async support
*   **Multi-tenant Architecture:** Project-based isolation with audit trails
*   **JSONL Persistence:** Tamper-evident storage with timestamp tracking
*   **Redis Caching:** Optional performance enhancement with graceful fallback

**🧪 TESTING STATUS:** 17/17 tests passing ✅
**🚀 DEPLOYMENT READY:** Production-ready with CI/CD pipeline
**🔧 CI/CD STATUS:** All jobs passing consistently ✅

### � World-Beater Hybrid Retrieval Engine 🔄 **IN DEVELOPMENT**

**Surpassing Claude/ChatGPT/Grok Capabilities:** Our 6-stage world-beater retrieval pipeline combines cutting-edge techniques to deliver superior contextual intelligence:

**Stage 1: Intent Analysis & Query Expansion**
- Advanced query understanding with multi-modal intent detection
- Dynamic query expansion using knowledge graph relationships
- Context-aware query reformulation for optimal retrieval

**Stage 2: Dense Retrieval Pipeline**
- Multi-vector dense retrieval with specialized encoders
- Hybrid semantic + lexical matching algorithms
- Real-time index updates with incremental learning

**Stage 3: Sparse Hybrid Integration**
- BM25 + semantic fusion with adaptive weighting
- Cross-encoder reranking for precision optimization
- Multi-stage filtering with confidence scoring

**Stage 4: Graph-Based Knowledge Expansion**
- Dynamic knowledge graph construction and traversal
- Entity relationship mining and expansion
- Contextual path finding with relevance weighting

**Stage 5: Neural Reranking & Personalization**
- Transformer-based cross-encoder reranking
- User behavior learning and personalization
- Multi-objective optimization for relevance and diversity

**Stage 6: Adaptive Learning & Optimization**
- Real-time performance monitoring and feedback loops
- Adaptive weight adjustment based on success metrics
- Continuous model improvement through reinforcement learning

**🔬 ADVANCED FEATURES:**
*   **Multi-Channel Delivery:** Push notifications, IDE integration, CLI alerts
*   **Real-Time Adaptation:** Dynamic pipeline optimization based on user feedback
*   **Enterprise Security:** End-to-end encryption with audit trails
*   **Performance Monitoring:** Comprehensive metrics and observability

**🧪 TESTING STATUS:** TDD test suites in development
**🚀 TARGET COMPLETION:** Q4 2025

### 🎯 Governance-Integrated Nudge Engine 🔄 **IN DEVELOPMENT**

**Proactive Behavioral Correction:** Advanced nudge system integrated with Parlant Governance for intelligent behavioral guidance:

**Core Intelligence:**
- **Behavioral Deviation Detection:** Real-time monitoring of agent behavior patterns
- **Contextual Nudge Generation:** Situation-aware guidance based on governance policies
- **Multi-Channel Delivery:** IDE notifications, CLI warnings, system alerts

**Governance Integration:**
- **Parlant Policy Engine:** Direct integration with governance frameworks
- **Compliance Monitoring:** Automated policy adherence verification
- **Ethical Safeguards:** Built-in safety mechanisms and override controls

**Adaptive Learning:**
- **User Response Analysis:** Learning from nudge effectiveness and user feedback
- **Dynamic Thresholds:** Adaptive sensitivity based on context and user preferences
- **Personalization Engine:** Individualized nudge strategies for optimal impact

**🔬 ADVANCED FEATURES:**
*   **Real-Time Policy Evaluation:** Continuous governance compliance monitoring
*   **Behavioral Pattern Recognition:** ML-powered anomaly detection
*   **Multi-Modal Communication:** Text, visual, and interactive nudge formats
*   **Audit Trail Integration:** Complete logging of all governance actions

**🧪 TESTING STATUS:** Governance integration tests in development
**🚀 TARGET COMPLETION:** Q4 2025

### �🔒 Security Auditor

Pinak integrates powerful security auditing capabilities directly into the developer workflow. It helps identify and mitigate security risks early in the development cycle.

*   **Secret Scanning:** Detect hardcoded secrets and sensitive information within your codebase.
*   **Dependency Vulnerability Scanning:** Identify known vulnerabilities in your project's dependencies.
*   **Comprehensive Reporting:** Generate detailed reports of security findings for quick remediation.

### 💻 CLI Interaction Logging

Gain unprecedented insight and auditability into your command-line activities. Pinak captures and stores every command executed, along with its context, enabling fast search, replay, and analysis of past sessions.

*   **Full Command Capture:** Records commands, arguments, timestamps, exit codes, and environment context.
*   **Privacy-First Redaction:** Automatically redacts sensitive information to ensure data privacy.
*   **Session Replay:** Replay past CLI sessions for debugging, learning, or auditing purposes.

### 💡 Pinakontext - SOTA Context Orchestrator

Pinakontext is the intelligence layer that brings all of Pinak's capabilities together, providing real-time, on-demand, and nudge-based context to developers and AI agents. It surfaces the *right* information at the *moment of need*.

*   **Realtime Context Delivery:** Get proactive tips and relevant information during critical development events (e.g., build failures, flaky tests, PR reviews).
*   **Hybrid Retrieval & Ranking:** Combines semantic and keyword search with intelligent reranking for highly relevant results.
*   **Policy-Driven Context:** Ensures that context delivery adheres to defined security and access policies.
*   **Extensible Recipes:** Define custom context scenarios using a flexible YAML-based DSL.

## Our Commitment to Excellence

We are committed to building Pinak with the highest standards:

*   **Test-Driven Development (TDD):** Ensuring robust and maintainable code.
*   **CI/CD Pipeline:** Automated testing, quality assurance, and continuous delivery with cross-platform compatibility.
*   **Cross-Platform Development:** Seamless development experience on macOS with CI/CD validation on Ubuntu.
*   **Platform-Specific Dependencies:** Smart dependency management for macOS-specific features.
*   **Regular GitHub Updates:** Maintaining a transparent and up-to-date codebase.
*   **Comprehensive Documentation:** Providing clear and accessible documentation for all features.
*   **No File Deletion Policy:** Critical files are archived, not deleted, ensuring data integrity.
*   **MacOS App Excellence:** Every design and implementation decision is made with the goal of a world-class macOS application.

## Getting Started

### Prerequisites
- **Python 3.13+** (Local development)
- **Python 3.11+** (CI/CD compatibility)
- **FAISS** (will be installed automatically)
- **Redis** (optional, for caching)
- **macOS** (for full feature set including menubar app)
- **Ubuntu/Cross-platform** (for CI/CD and core functionality)

### Platform-Specific Features

**macOS (Full Feature Set):**
- Complete menubar application (`rumps` framework)
- System integration via `pyobjc`
- All memory service features
- CLI interaction logging

**Cross-Platform (Core Features):**
- Memory service API (17 endpoints)
- CLI tools and commands
- All testing and CI/CD functionality
- Web-based interfaces

### Quick Start - Memory Service

1. **Clone and Setup:**
```bash
git clone https://github.com/Pinak-Setu/Pinak_Projects.git
cd Pinak_Projects/Pinak_Services/memory_service
```

2. **Install Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the Service:**
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. **Test the API:**
```bash
# Add semantic memory
curl -X POST "http://localhost:8000/api/v1/memory/add" \
  -H "Content-Type: application/json" \
  -d '{"content": "Python is great for AI development", "tags": ["python", "ai"]}'

# Search memories
curl "http://localhost:8000/api/v1/memory/search?query=python"
```

5. **Run Tests:**
```bash
python -m pytest tests/ -v
```

### API Documentation
Once running, visit `http://localhost:8000/docs` for interactive API documentation.

### Memory Layers Available:
- **Semantic**: Vector-based similarity search
- **Episodic**: Personal experiences with salience scoring
- **Procedural**: Step-by-step instructions and skills
- **RAG**: External knowledge integration
- **Events**: Audit trail and activity logging
- **Session**: Temporary context with TTL
- **Working**: Scratch memory for immediate tasks
- **Changelog**: Version history and redaction

## Contributing

We welcome contributions from the community. Please refer to our `CONTRIBUTING.md` for guidelines.

## License

*(License information will be provided here.)*

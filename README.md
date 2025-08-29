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

### 🔒 Security Auditor

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
*   **CI/CD Pipeline:** For automated testing, quality assurance, and continuous delivery.
*   **Regular GitHub Updates:** Maintaining a transparent and up-to-date codebase.
*   **Comprehensive Documentation:** Providing clear and accessible documentation for all features.
*   **No File Deletion Policy:** Critical files are archived, not deleted, ensuring data integrity.
*   **MacOS App Excellence:** Every design and implementation decision is made with the goal of a world-class macOS application.

## Getting Started

### Prerequisites
- Python 3.13+
- FAISS (will be installed automatically)
- Redis (optional, for caching)

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

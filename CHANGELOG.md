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


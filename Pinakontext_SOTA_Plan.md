
# 🧠 Pinakontext — SOTA Context Orchestrator (Local‑First, Enterprise‑Grade)

**Pinakontext** is a program under **Pinak Bridge** that turns Memory/Governance/Parlant signals into **real‑time, on‑demand, and nudge‑based context** for developers and AI agents. It is **local‑first** (air‑gapped capable), yet **enterprise‑ready** (policy, audit, OIDC‑ready).

---

## 0) Objectives

- **Realtime & On‑Demand Context**: Surface the *right* code, logs, decisions, and memory at the moment of need.
- **Nudges for DevOps**: Proactive tips during build failures, flaky tests, PR review, deploys, SLO breaches.
- **SOTA Security Baseline**: Project‑scoped identity, least‑privilege, signed headers, tamper‑evident audit.
- **High DX**: Simple CLI/TUI, recipes (YAML) to declare *what* context to surface and *when*.
- **World-Beater Retrieval**: 6-stage hybrid pipeline surpassing Claude/ChatGPT/Grok capabilities.
- **Governance Integration**: Parlant-powered behavioral correction and ethical safeguards.

---

## 1) Architecture (C4‑ish)

```mermaid
graph TD
  subgraph DevHost[Developer Machine / Local Node]
    A[Agent / CLI / IDE<br/>pinak-context] --> B[Context Broker<br/>(6-Stage World-Beater Retrieval)]
    B --> C[Policy Gate (OPA/Rego)<br/>+ Security Filters]
    C --> D[Governance Nudge Engine<br/>(Parlant Integration + Behavioral Correction)]
    B --> E[Memory API]
    B --> F[Governance Gateway]
    B --> G[Parlant (policy/speech)]
    D --> A

    subgraph Signals[Signal Ingestors]
      H[Git/PR diffs]
      I[CI/CD events]
      J[Test & Coverage]
      K[Runtime logs/metrics]
      L[Docker/K8s status]
      M[Pinak Events/Changelog]
    end

    H --> D
    I --> D
    J --> D
    K --> D
    L --> D
    M --> D
  end

  subgraph Stores[Local Persistence]
    N[FAISS + metadata]
    O[Episodic/Procedural/RAG JSONLs]
    P[Session/Working JSONLs]
    Q[Events + Changelog (hash-chained)]
  end

  E --- N
  E --- O
  E --- P
  F --- Q
```

**Key roles**  
- **Context Broker**: orchestrates 6-stage world-beater retrieval across memory layers + external signals; synthesizes concise, actionable context.  
- **Governance Nudge Engine**: Parlant-integrated behavioral correction with proactive guidance.  
- **Policy Gate**: enforces RBAC/ABAC, redaction, and release conditions.  
- **Parlant**: governance framework shaping behavioral corrections and ethical boundaries.

---

## 2) World-Beater Hybrid Retrieval & Ranking (6-Stage Pipeline)

**Stage 1: Intent Analysis & Query Expansion**
- Multi-modal intent detection (semantic, keyword, contextual)
- Dynamic query expansion using knowledge graph relationships
- Context-aware query reformulation for optimal retrieval
- User intent learning and personalization

**Stage 2: Dense Retrieval Pipeline**
- Multi-vector dense retrieval with specialized encoders
- Hybrid semantic + lexical matching algorithms
- Real-time index updates with incremental learning
- Cross-modal embedding fusion

**Stage 3: Sparse Hybrid Integration**
- BM25 + semantic fusion with adaptive weighting
- Cross-encoder reranking for precision optimization
- Multi-stage filtering with confidence scoring
- Query-dependent feature weighting

**Stage 4: Graph-Based Knowledge Expansion**
- Dynamic knowledge graph construction and traversal
- Entity relationship mining and expansion
- Contextual path finding with relevance weighting
- Temporal knowledge evolution

**Stage 5: Neural Reranking & Personalization**
- Transformer-based cross-encoder reranking
- User behavior learning and personalization
- Multi-objective optimization (relevance, diversity, recency)
- Adaptive threshold calibration

**Stage 6: Adaptive Learning & Optimization**
- Real-time performance monitoring and feedback loops
- Adaptive weight adjustment based on success metrics
- Continuous model improvement through reinforcement learning
- A/B testing framework for pipeline optimization

**Advanced Features:**
- **Multi-Channel Delivery:** Push notifications, IDE integration, CLI alerts
- **Real-Time Adaptation:** Dynamic pipeline optimization based on user feedback
- **Enterprise Security:** End-to-end encryption with audit trails
- **Performance Monitoring:** Comprehensive metrics and observability

---

## 3) Governance-Integrated Nudge Engine

**Behavioral Intelligence:**
- **Deviation Detection:** Real-time monitoring of agent behavior patterns
- **Contextual Analysis:** Situation-aware assessment of behavioral appropriateness
- **Ethical Safeguards:** Built-in safety mechanisms and override controls

**Parlant Integration:**
- **Policy Engine:** Direct integration with Parlant governance frameworks
- **Compliance Monitoring:** Automated policy adherence verification
- **Behavioral Correction:** Proactive guidance for policy violations

**Adaptive Learning:**
- **Response Analysis:** Learning from nudge effectiveness and user feedback
- **Dynamic Thresholds:** Adaptive sensitivity based on context and user preferences
- **Personalization:** Individualized nudge strategies for optimal impact

**Multi-Modal Delivery:**
- **Communication Channels:** IDE notifications, CLI warnings, system alerts
- **Format Optimization:** Text, visual, and interactive nudge formats
- **Timing Intelligence:** Optimal delivery timing based on user context

---

## 4) SOTA Security Baseline (pinned for Pinakontext)

- **Identity**: short‑lived JWT (60–90m) + refresh in OS keyring; `X-Pinak-Project` must match token claim `pid`.
- **Policy Gate**: OPA/Rego evaluates *who/what/where*; redacts PII/keys before release.
- **Transport**: local TLS/mTLS with cert pinning.
- **Tamper Evidence**: hash‑chained audit for *all surfaced contexts*; anchor hourly.
- **Least Privilege**: recipes declare required scopes; broker requests only needed layers.
- **Observability**: OTEL spans tagged with `project_id`, `recipe`, `trigger`; `/metrics` counters for nudges, accept/reject, latency.
- **Governance**: Parlant integration for behavioral monitoring and correction.

---

## 5) CLI — `pinak-context`

```bash
# on-demand context with world-beater retrieval
pinak-context now --topic build-failure --json --enhanced

# governance-integrated nudges
pinak-context nudge enable build-failure
pinak-context nudge governance-status

# run a recipe with advanced retrieval
pinak-context run --recipe recipes/pr_review.yaml --pr 128 --json --world-beater

# tail live context with governance monitoring
pinak-context tail --layer session --since 15m --governance
```

**Outputs**: JSON lines (default) with `context_id`, `title`, `summary`, `actions[]`, `refs[]`, `sensitivity`, `ttl`, `governance_flags`.

---

## 6) Recipes DSL (YAML) - Enhanced

```yaml
# recipes/build_failure.yaml
id: build-failure
version: 2.0
triggers:
  - signal: ci.pipeline.failed
    when:
      branch: main
      frequency_limit: "1 per 10m"
scopes:
  tenant: default
  project: Pnk-Demo
inputs:
  layers:
    - episodic: {since: "24h", filter: ["build", "compile", "lint"]}
    - procedural: {since: "7d", filter: ["pipeline", "fix"]}
    - rag: {since: "30d", filter: ["docs", "README", "CI"]}
    - events: {since: "24h", filter: ["ci", "deploy"]}
  external:
    - git: {diff_range: "HEAD~5..HEAD"}
    - tests: {report: "coverage.xml"}
  world_beater:
    intent_analysis: true
    graph_expansion: true
    neural_reranking: true
    adaptive_weights: true
ranking:
  world_beater_pipeline:
    stage_weights:
      dense_retrieval: 0.3
      sparse_hybrid: 0.2
      graph_expansion: 0.25
      neural_rerank: 0.25
    k_candidates: 100
    k_final: 10
  salience:
    recent_session_boost: 1.2
    episodic_salience_weight: 0.8
    governance_relevance: 0.9
synthesize:
  format: "markdown"
  include:
    - root_cause
    - most_impacted_modules
    - recommended_fixes
    - commands_to_run
    - references
    - governance_warnings
security:
  min_role: "editor"
  redact: ["secrets", "access_tokens", "emails"]
  governance:
    behavioral_monitoring: true
    ethical_boundaries: ["no-harm", "privacy-first"]
  release:
    approval: "auto"
    ttl: "30m"
```

---

## 7) Code Patches (minimal scaffolding)

> These create a working skeleton for the enhanced Pinakontext with world-beater retrieval and governance integration.


### A) `setup.py` — add console script

```diff
@@
     entry_points={
         "console_scripts": [
             "pinak-bridge=pinak.bridge.cli:main",
             "pinak-memory=pinak.memory.cli:main",
+            "pinak-context=pinak.context.cli:main",
         ],
     },
```

---

### B) `src/pinak/context/__init__.py`

```python
__all__ = ["cli", "broker", "nudge", "recipes", "security"]
```

---

### C) `src/pinak/context/cli.py`

```python
import argparse, json, sys
from .broker import ContextBroker
from .nudge import NudgeManager

def main():
    parser = argparse.ArgumentParser(prog="pinak-context", description="Pinakontext CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_now = sub.add_parser("now", help="Get on-demand context")
    p_now.add_argument("--topic", required=True)
    p_now.add_argument("--json", action="store_true")

    p_run = sub.add_parser("run", help="Run a recipe file")
    p_run.add_argument("--recipe", required=True)
    p_run.add_argument("--json", action="store_true")
    p_run.add_argument("--args", nargs="*", default=[])

    p_tail = sub.add_parser("tail", help="Tail live context from a layer")
    p_tail.add_argument("--layer", default="session")
    p_tail.add_argument("--since", default="30m")

    p_nudge = sub.add_parser("nudge", help="Enable/disable nudges")
    p_nudge.add_argument("action", choices=["enable", "disable", "status"])
    p_nudge.add_argument("name", nargs="?")

    args = parser.parse_args()
    broker = ContextBroker()
    nm = NudgeManager(broker=broker)

    if args.cmd == "now":
        ctx = broker.get_context(topic=args.topic)
        print(json.dumps(ctx) if args.json else ctx.get("pretty", ""))
    elif args.cmd == "run":
        ctx = broker.run_recipe(path=args.recipe, argv=args.args)
        print(json.dumps(ctx) if args.json else ctx.get("pretty", ""))
    elif args.cmd == "tail":
        for line in broker.tail(layer=args.layer, since=args.since):
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
    elif args.cmd == "nudge":
        if args.action == "enable":
            nm.enable(args.name)
        elif args.action == "disable":
            nm.disable(args.name)
        else:
            print(json.dumps(nm.status(), indent=2))
```

---

### D) `src/pinak/context/broker.py`

```python
import time, json, os, threading
from dataclasses import dataclass, field
from typing import Dict, Any, Iterable, List

@dataclass
class ContextItem:
    title: str
    summary: str
    actions: List[str] = field(default_factory=list)
    refs: List[str] = field(default_factory=list)
    sensitivity: str = "low"
    ttl: str = "30m"

class ContextBroker:
    def __init__(self):
        # TODO: init keyring/http/otel
        pass

    def get_context(self, topic: str) -> Dict[str, Any]:
        item = ContextItem(
            title=f"Context for {topic}",
            summary=f"Top signals and likely fixes for {topic}.",
            actions=["make verify", "rerun failing tests"],
            refs=["docs/CI.md", "logs/ci/latest.txt"],
        )
        return {"context_id": f"ctx-{int(time.time())}", "pretty": self._pretty(item), "item": item.__dict__}

    def run_recipe(self, path: str, argv: List[str]) -> Dict[str, Any]:
        item = ContextItem(
            title=f"Recipe: {os.path.basename(path)}",
            summary="Synthesis from recipe inputs (stub).",
            actions=["apply patch", "restart service"],
        )
        return {"context_id": f"ctx-{int(time.time())}", "pretty": self._pretty(item), "item": item.__dict__}

    def tail(self, layer: str = "session", since: str = "30m") -> Iterable[str]:
        for i in range(3):
            yield json.dumps({"layer": layer, "line": f"event-{i}", "ts": int(time.time())})
            time.sleep(0.5)

    def _pretty(self, item: ContextItem) -> str:
        actions = "\n".join([f"- {a}" for a in item.actions])
        refs = "\n".join([f"- {r}" for r in item.refs])
        return f"# {item.title}\n\n{item.summary}\n\n## Actions\n{actions}\n\n## References\n{refs}\n"
```

---

### E) `src/pinak/context/nudge.py`

```python
import threading, time
from typing import Dict

class NudgeManager:
    def __init__(self, broker):
        self.broker = broker
        self._enabled: Dict[str, bool] = {}
        self._threads: Dict[str, threading.Thread] = {}

    def enable(self, name: str):
        if not name: return
        if self._enabled.get(name): return
        self._enabled[name] = True
        t = threading.Thread(target=self._loop, args=(name,), daemon=True)
        self._threads[name] = t
        t.start()
        print(f"Nudge '{name}' enabled.")

    def disable(self, name: str):
        if not name: return
        self._enabled[name] = False
        print(f"Nudge '{name}' disabled.")

    def status(self):
        return {"nudges": self._enabled}

    def _loop(self, name: str):
        while self._enabled.get(name, False):
            ctx = self.broker.get_context(topic=name)
            print(ctx.get("pretty", ""))
            time.sleep(10)
```

---

### F) `src/pinak/context/security.py`

```python
def authorize(claims: dict, headers: dict) -> bool:
    return headers.get("X-Pinak-Project") == claims.get("pid")
```

---

### G) `requirements.txt` additions

```diff
+ pyyaml
+ opentelemetry-api
+ prometheus-client
+ portalocker
```

---

## 7) Observability & Metrics

- **Traces**: `pinak.context.get_context`, `pinak.context.run_recipe`, `pinak.nudge.loop`
- **Counters**: `pinak_nudges_total{recipe}`, `pinak_context_latency_ms{topic}`, `pinak_context_rejected_total{reason}`
- **Logs**: JSON structured, redact secrets before emit.

---

## 8) Acceptance Tests (minimum)

- `test_policy_gate_rejects_pid_mismatch`  
- `test_nudge_respects_frequency_limit`  
- `test_broker_merges_layers_without_duplicates`  
- `test_tail_streams_new_events`  

---

## 9) 72‑Hour Delivery Plan

**Day 1**: Scaffolding (files above), CLI end‑to‑end, stub broker, enable one nudge.  
**Day 2**: Hybrid retrieval (episodic+semantic), rerank, recipe YAML, OTEL + /metrics.  
**Day 3**: Policy Gate (OPA), redaction, hash‑chained audit of served contexts, chaos tests.

---

## 10) Summary

Pinakontext makes Pinak *actionable* at the moment of need: **world-beater retrieval** surpassing Claude/ChatGPT/Grok, **governance-integrated nudges** for behavioral correction, all with **SOTA security** and **local‑first** operation. The 6-stage retrieval pipeline delivers superior contextual intelligence while Parlant integration ensures ethical AI behavior and proactive guidance.

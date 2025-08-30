# 🏹 Pinak — Enterprise-grade Reference Architecture (Local-First, SOTA Secure)

This document captures the **enterprise-grade reference architecture** for **Pinak**,
including **SOTA security baseline** and **code patch examples** for practical integration.

---

## 1) Architecture Diagram

```mermaid
graph TD
  subgraph DevHost[Developer Machine / Local Node]
    A[Agent / App<br/>SDK + MemoryManager] --> B[Pinak Bridge<br/>(Context + Keyring + Token Flow)]
    B -->|OIDC (optional) or Dev JWT| C[Token Service<br/>(Local OIDC/Dev Issuer)]
    B -->|AuthZ headers| D[Governance Gateway (Parlant Integration)]
    A -->|/api/v1/memory/*| E[Memory API]
    A -->|6-Stage World-Beater Retrieval| F[Context Broker<br/>(Advanced Retrieval Pipeline)]
    A -->|Behavioral Correction| G[Governance Nudge Engine]

    subgraph Storage[Local Persistence (Project & Tenant Scoped)]
      H[Semantic Index<br/>FAISS + Graph DB + metadata.json]
      I[Episodic.jsonl (time-partitioned)]
      J[Procedural.jsonl (time-partitioned)]
      K[RAG.jsonl (time-partitioned)]
      L[Session_*.jsonl]
      M[Working.jsonl]
      N[Events.jsonl (hash-chained)]
      O[Changelog.jsonl (hash-chained)]
    end
    E --- H
    E --- I
    E --- J
    E --- K
    E --- L
    E --- M
    E --- N
    E --- O

    subgraph Obs[Observability]
      P[OTEL Traces/Logs<br/>(project_id, tenant, layer)]
      Q[Prometheus /metrics]
    end

    E --> P
    E --> Q
    D --> P
    F --> P
    G --> P
  end

  style Storage fill:#0b1220,stroke:#6aa,stroke-width:1px,color:#dff
  style DevHost fill:#0b0f17,stroke:#aaa,stroke-width:1px,color:#fff
```

---

## 2) World-Beater Hybrid Retrieval Pipeline

### 6-Stage Advanced Retrieval Architecture

**Stage 1: Intent Analysis & Query Expansion**
- Multi-modal intent detection using transformer models
- Dynamic query expansion with knowledge graph traversal
- Context-aware query reformulation for optimal retrieval
- User intent learning and personalization algorithms

**Stage 2: Dense Retrieval Pipeline**
- Multi-vector dense retrieval with specialized sentence encoders
- Hybrid semantic + lexical matching with adaptive algorithms
- Real-time index updates with incremental learning
- Cross-modal embedding fusion techniques

**Stage 3: Sparse Hybrid Integration**
- BM25 + semantic fusion with dynamic weighting
- Cross-encoder reranking for precision optimization
- Multi-stage filtering with confidence scoring
- Query-dependent feature weighting optimization

**Stage 4: Graph-Based Knowledge Expansion**
- Dynamic knowledge graph construction using Neo4j/GraphDB
- Entity relationship mining and expansion algorithms
- Contextual path finding with relevance weighting
- Temporal knowledge evolution tracking

**Stage 5: Neural Reranking & Personalization**
- Transformer-based cross-encoder reranking models
- User behavior learning and personalization engines
- Multi-objective optimization (relevance, diversity, recency)
- Adaptive threshold calibration systems

**Stage 6: Adaptive Learning & Optimization**
- Real-time performance monitoring and feedback loops
- Adaptive weight adjustment based on success metrics
- Continuous model improvement through reinforcement learning
- A/B testing framework for pipeline optimization

---

## 3) Governance-Integrated Nudge Engine

### Behavioral Intelligence & Correction

**Core Intelligence:**
- **Deviation Detection:** Real-time monitoring of agent behavior patterns using ML
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

## 4) Data Layout (Partitioned JSONL)

```
Pinak_Services/memory_service/data/
  memory.faiss
  metadata.json
  knowledge_graph.db

  tenants/<tenant>/<project>/
    episodic/episodic_YYYY-MM-DD.jsonl
    procedural/procedural_YYYY-MM-DD.jsonl
    rag/rag_YYYY-MM-DD.jsonl
    session/session_<id>.jsonl
    working/working.jsonl
    events/events_YYYY-MM-DD.jsonl
    changelog/changes_YYYY-MM-DD.jsonl
    governance/governance_YYYY-MM-DD.jsonl
```

---

## 5) SOTA Security Baseline

- **Identity & Auth**: short-lived JWTs, refresh, OIDC optional
- **Storage**: JSONL is time-partitioned, append-only, file-locked
- **Audit**: hash-chained logs, anchored hourly
- **Observability**: OTEL traces, Prometheus metrics
- **Secrets**: OS keyring, fallback with 0600 perms only
- **Transport**: TLS/mTLS with pinned certs
- **Governance**: Parlant integration for behavioral monitoring
- **Retrieval**: End-to-end encryption for world-beater pipeline---

## 4) Code Patches

### 🔒 Add File Locking for JSONL Writes

**File: `manager.py`**
```python
import portalocker

def append_jsonl(file_path: str, entry: dict):
    with open(file_path, "a") as f:
        with portalocker.Lock(f, timeout=5):
            f.write(json.dumps(entry) + "\n")
```

---

### 🔐 Add Hash-Chained Audit Entries

**File: `auditor.py`**
```python
import hashlib, json, time

def compute_entry_hash(entry: dict, prev_hash: str = "0"):
    base = json.dumps(entry, sort_keys=True)
    return hashlib.sha256((prev_hash + base).encode()).hexdigest()

def append_audit(file_path: str, entry: dict, prev_hash: str):
    entry["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    entry["prev_hash"] = prev_hash
    entry["entry_hash"] = compute_entry_hash(entry, prev_hash)
    with open(file_path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry["entry_hash"]
```

---

### 🔑 Bridge Token Rotation

**File: `bridge/context.py`**
```python
def rotate_token(project_id: str):
    # Placeholder: in enterprise, integrate with OIDC
    new_token = issue_new_dev_jwt(project_id)
    keyring.set_password("pinak", f"project:{project_id}", new_token)
    return new_token
```

---

### 📊 Observability Hooks

**File: `memory_service/app/main.py`**
```python
from opentelemetry import trace
from prometheus_client import Counter

trace.set_tracer_provider(...)
tracer = trace.get_tracer("pinak.memory")

REQUEST_COUNTER = Counter("pinak_requests", "Requests per layer", ["layer", "project_id"])

@app.post("/api/v1/memory/{layer}/add")
def add_memory(layer: str, request: Request):
    with tracer.start_as_current_span(f"memory.add.{layer}") as span:
        pid = request.headers.get("X-Pinak-Project")
        REQUEST_COUNTER.labels(layer=layer, project_id=pid).inc()
        ...
```

---

## 5) Makefile Targets

```makefile
demo:
	uv run -m pinak.scripts.dev_issuer &
	uvicorn app.main:app --port 8011 --reload &
	pinak-bridge init --name Demo --url http://localhost:8011 --tenant default
	pinak-bridge verify
	pinak-memory health

verify:
	gitleaks detect --no-banner || true
	semgrep --error --config auto || true
	pinak-bridge verify
```

---

## 6) Policy Example (OPA/Rego)

```rego
package pinak.policy

default allow = false

allow {
  input.claims.role == "editor"
  input.headers["x-pinak-project"] == input.claims.pid
  input.request.path == "/api/v1/memory/episodic/add"
}
```

---

## ✅ Next Steps

1. Add **short-lived JWT refresh flow**
2. Time-partition JSONLs + nightly compaction to Parquet
3. Hash-chain events & changelog with hourly anchors
4. Enforce **file-locking** everywhere
5. Ship `/metrics` + OTEL spans

---

## ✅ Acceptance Checklist (Executable)

- One-click orchestration
  - `pinak up` runs security preflight and brings up services.
  - `pinak status` shows Memory API healthy.
  - `pinak down` stops containers.

- Identity + rotation
  - `pinak token --exp 30 --set` mints a short-lived JWT and stores it.
  - `pinak-bridge token rotate --exp 60 --role editor` refreshes token.

- 8-layer memory verification
  - Run: `python3 Pinak_Package/scripts/demo_all_layers.py`
  - Verify files under `Pinak_Services/memory_service/data/tenants/<tenant>/<project>/`:
    - episodic/episodic_YYYY-MM-DD.jsonl
    - procedural/procedural_YYYY-MM-DD.jsonl
    - rag/rag_YYYY-MM-DD.jsonl
    - session/session_<id>.jsonl
    - working/working.jsonl
    - events/events_YYYY-MM-DD.jsonl (hash-chained)
    - changelog/changes_YYYY-MM-DD.jsonl (hash-chained)
    - semantic index: data/memory.faiss + data/metadata.json

- Observability
  - Metrics: ensure `PINAK_METRICS=true` (compose sets it). Hit `/metrics` and see `pinak_requests_total`.
  - Tracing: set `PINAK_OTEL=true` to emit dev spans (ConsoleSpanExporter) in logs.

- Security & image policy
  - Containers run as non-root user, with read-only root FS; `/data` is a writable volume.
  - CI gates enforce secrets scanning, SAST critical=0, KEVs=0.

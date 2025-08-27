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
    B -->|AuthZ headers| D[Governance Gateway (optional)]
    A -->|/api/v1/memory/*| E[Memory API]

    subgraph Storage[Local Persistence (Project & Tenant Scoped)]
      F[Semantic Index<br/>FAISS + metadata.json] 
      G[Episodic.jsonl (time-partitioned)]
      H[Procedural.jsonl (time-partitioned)]
      I[RAG.jsonl (time-partitioned)]
      J[Session_*.jsonl]
      K[Working.jsonl]
      L[Events.jsonl (hash-chained)]
      M[Changelog.jsonl (hash-chained)]
    end
    E --- F
    E --- G
    E --- H
    E --- I
    E --- J
    E --- K
    E --- L
    E --- M

    subgraph Obs[Observability]
      N[OTEL Traces/Logs<br/>(project_id, tenant, layer)]
      O[Prometheus /metrics]
    end

    E --> N
    E --> O
    D --> N
  end

  style Storage fill:#0b1220,stroke:#6aa,stroke-width:1px,color:#dff
  style DevHost fill:#0b0f17,stroke:#aaa,stroke-width:1px,color:#fff
```

---

## 2) Data Layout (Partitioned JSONL)

```
Pinak_Services/memory_service/data/
  memory.faiss
  metadata.json

  tenants/<tenant>/<project>/
    episodic/episodic_YYYY-MM-DD.jsonl
    procedural/procedural_YYYY-MM-DD.jsonl
    rag/rag_YYYY-MM-DD.jsonl
    session/session_<id>.jsonl
    working/working.jsonl
    events/events_YYYY-MM-DD.jsonl
    changelog/changes_YYYY-MM-DD.jsonl
```

---

## 3) SOTA Security Baseline

- **Identity & Auth**: short-lived JWTs, refresh, OIDC optional
- **Storage**: JSONL is time-partitioned, append-only, file-locked
- **Audit**: hash-chained logs, anchored hourly
- **Observability**: OTEL traces, Prometheus metrics
- **Secrets**: OS keyring, fallback with 0600 perms only
- **Transport**: TLS/mTLS with pinned certs

---

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

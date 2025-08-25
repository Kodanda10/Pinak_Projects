# Iron-Clad 2025 Agent Memory Blueprint

## 1. Canonical Layers
1. **Working** – Volatile scratchpad (ms–seconds).
2. **Session** – Short-term per thread, LangGraph checkpointers distill to LTM.
3. **Event** – Immutable log of tool calls, states, decisions.
4. **Changelog** – Append-only diffs, hash-chained, WORM.
5. **Episodic** – Past experiences, outcomes, reflections.
6. **Semantic** – Persistent facts, KBs, knowledge graphs.
7. **Procedural** – Learned skills, workflows, tool plans.
8. **RAG/External** – Hybrid retrieval + agentic RAG from outside KBs/web.

## 2. JSON Schema (Base + Extensions)
- **Base fields**: id, content, embedding, timestamp, tags, metadata, hash.
- **Working**: add relevance_score.
- **Session**: add session_id, parent_id.
- **Event**: add event_type, outcome.
- **Changelog**: add change_type, actor.
- **Semantic**: add source, confidence.
- **Procedural**: add steps, success_rate, reflection_loop: boolean (for self-refinement).
- **RAG**: add external_source, relevance_score.

All entries must carry vector embeddings → unified semantic retrieval across layers.

## 3. TTL & Eviction
- Working: 1–5 min, LRU.
- Session: 1–24h, expire on close (distill to LTM).
- Event: 30–90d, evict low-salience (<0.5).
- Changelog: Permanent (WORM).
- Semantic/Procedural: Indefinite; evict <0.6 confidence or <5/year access.
- RAG cache: 5–30 min.
- Global: <70% utilization; hybrid LRU/LFU weighted by salience; excessive data eviction if CPR >5%.

## 4. Read/Write & Conflict Resolution
- Append-only for Event/Changelog.
- Versioned + optimistic locking for Semantic.
- Procedural updates ranked by success_rate.
- Multi-agent: default CRDT (Automerge); fallback LWW timestamp; log resolution to Changelog; human-in-loop for unresolved.

## 5. Retrieval Ranking
- Hybrid candidate gen: dense (ANN) + sparse (BM25/SPLADE).
- Combine via RRF (Reciprocal Rank Fusion).
- Add recency boost + confidence weighting.
- Re-rank top-20 → top-5 with cross-encoders (MonoT5, BGE).
- Multi-modal embeddings (text+image+audio) for RAG.
- Personalize with tags/user context.

Formula:  
score = 0.45*dense + 0.25*sparse + 0.20*recency + 0.10*confidence

## 6. Storage
- Working/Session → Redis.
- Event/Changelog → Kafka + S3 Object Lock (WORM).
- Episodic/Semantic → Milvus/Pinecone (multi-vector) + MongoDB/Neo4j.
- Procedural → KV/Git-style versioning.
- Hybrid edge+cloud for privacy/scale; optional IPFS/Filecoin + on-chain L1 for decentralized MAS.

## 7. Multi-Agent Sync
- Pub/sub (Kafka).
- CRDT deltas broadcast; arbiter fallback.
- Differential sync to cut bandwidth.
- State consistency lag <2s.
- Fallback protocols for errors (retry, human escalation).

## 8. Privacy & Governance
- PII scrub → pseudonymisation vaults (EDPB 2025).
- DP calibration (NIST SP 800-226).
- RBAC + OPA policies.
- WORM + hash-chain changelog.
- OpenLineage for lineage exposure.
- User-facing “forget me” API.
- Reputation scoring per agent (success_rate + trust metrics).

## 9. KPIs
- Retrieval accuracy ≥95%.
- LoCoMo ≥80% long-term consistency.
- LongMemEval ≥85% AR/TTL/LRU/CR.
- Retrieval latency <500ms p95.
- CPR <5%.
- Memory hit rate ≥80%.
- PII leakage = 0.
- Task success lift: 25-40%.

## 10. Guardrails
- Memory bloat → salience pruning.
- Privacy leaks → DP + vaults.
- Sync deadlocks → CRDT + timeouts.
- Retrieval bias → RRF + A/B test.
- RAG overuse → cache + hybrid thresholds.
- Vendor lock-in → multi-model compatibility.

## 11. Config (memory.json)
{
  "layers": {
    "working": {"ttl_seconds": 300},
    "session": {"ttl_seconds": 7200, "distill_on_close": true},
    "event": {"ttl_seconds": 2592000},
    "changelog": {"ttl_seconds": 0},
    "semantic": {"ttl_seconds": 0, "review_days": 90},
    "procedural": {"ttl_seconds": 0},
    "rag": {"ttl_seconds": 600}
  },
  "retrieval": {
    "fusion": "RRF",
    "weights": {"dense":0.45,"sparse":0.25,"rerank":0.20,"recency":0.10},
    "multi_vector": true
  },
  "safety": {"human_loop_threshold": 0.8}
}

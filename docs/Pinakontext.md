# Pinakontext - SOTA Context Orchestrator

## Overview

**Pinakontext** is a pivotal new program under the Pinak Bridge, designed to transform how developers and AI agents interact with information. It acts as a State-of-the-Art (SOTA) Context Orchestrator, turning various signals (from memory, governance, and other sources) into **real-time, on-demand, and nudge-based context**.

Built with a **local-first** approach (air-gapped capable), Pinakontext is simultaneously **enterprise-ready**, incorporating robust policy enforcement, auditing capabilities, and OIDC integration. Its core objective is to surface the *right* code, logs, decisions, and memory at the moment of need, providing proactive tips and insights.

## Key Objectives

*   **Realtime & On-Demand Context**: Deliver relevant information precisely when and where it's needed.
*   **Nudges for DevOps**: Provide proactive guidance during critical development events (e.g., build failures, flaky tests, PR reviews, deploys, SLO breaches).
*   **SOTA Security Baseline**: Ensure project-scoped identity, least-privilege access, signed headers, and tamper-evident audit trails.
*   **High Developer Experience (DX)**: Offer a simple CLI/TUI and intuitive YAML-based recipes to declare context requirements.

## Architecture Highlights

Pinakontext operates on a developer machine/local node, orchestrating context through key components:

*   **Context Broker**: Orchestrates retrieval across various memory layers and external signals, synthesizing concise and actionable context.
*   **Nudge Engine**: A rule/ML-driven system that triggers proactive tips based on events like build failures or test flakiness.
*   **Policy Gate**: Enforces Role-Based Access Control (RBAC) and Attribute-Based Access Control (ABAC), performs redaction, and manages release conditions.
*   **Parlant (Optional)**: A policy/speech layer that shapes the final wording or guidance persona.

It ingests signals from diverse sources including Git/PR diffs, CI/CD events, Test & Coverage reports, Runtime logs/metrics, Docker/K8s status, and Pinak Events/Changelog.

## Retrieval & Ranking (SOTA)

Pinakontext employs advanced retrieval and ranking techniques:

*   **Hybrid Retrieval**: Combines BM25/sparse search with semantic (FAISS) and rule-based filters.
*   **Reranking**: Utilizes local cross-encoders and salience from memory layers.
*   **Context Graph**: Maintains a lightweight graph linking various development artifacts (files, tests, incidents, commits, memories).
*   **Answer Synthesis**: Structures output with outlines, actionable commands, references, and safety notes.
*   **Caching**: Implements request-scoped caching and hot-set prefetching.

## SOTA Security Baseline

Security is paramount, with features including:

*   **Identity**: Short-lived JWTs with refresh mechanisms.
*   **Policy Gate**: OPA/Rego for granular access control and PII/key redaction.
*   **Transport**: Local TLS/mTLS with cert pinning.
*   **Tamper Evidence**: Hash-chained audit for all surfaced contexts.
*   **Least Privilege**: Recipes declare required scopes, ensuring minimal access.
*   **Observability**: OTEL spans and Prometheus counters for security-related metrics.

## CLI (`pinak-context`)

Pinakontext provides a dedicated CLI for on-demand context, managing nudges, running recipes, and tailing live context. Outputs are typically JSON lines, providing structured information.

## Recipes DSL (YAML)

Context requirements are defined using a flexible YAML-based Domain Specific Language (DSL), allowing for precise control over triggers, scopes, inputs, ranking, synthesis, and security policies.

## Further Details

For a comprehensive understanding of Pinakontext's design, architecture, and implementation details, please refer to the full plan document: [`Pinakontext_SOTA_Plan.md`](../Pinakontext_SOTA_Plan.md)

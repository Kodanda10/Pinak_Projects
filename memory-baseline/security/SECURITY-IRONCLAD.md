# The Ironclad Blueprint: Architecting a World‑Class Code Security Audit System (2025 Edition, v3)
_Last updated: 2025-08-25 16:24 IST_

> **What’s new in v3 (Aug 2025)**  
> 2025-standard updates across **OWASP ASVS 5.0 (final)**, **NIST SSDF v1.1 + SP 800‑218A guidance**, **SLSA v1.1** provenance, **Zero‑Trust CI/CD**, **eBPF** runtime monitoring, **LLM/AI security testing**, **crypto‑agility + PQC pilots**, **ASPM with EPSS/KEV risk correlation**, **BAS** validation, hardened **quality gates** (sample in `security/policy/ci-security-gates.yaml`), and India hooks (**DPDP Act** + **CERT‑In 6‑hour IR**).

---

## 0) Anchor Standards & Alignment Targets

- **OWASP ASVS 5.0** — baseline control catalogue + release gates.  
  **Target**: Tier‑0 apps min **L2**, payment/PII/critical **L3**.
- **NIST SSDF v1.1 (SP 800‑218)** + **SP 800‑218A** — program scaffold (policy, roles, checkpoints). Bake into backlog templates and release checklists.
- **SLSA v1.1** — provenance on all build artifacts, hermetic/isolated builders, attestations verified in CD.
- **CVSS v4.0 + EPSS + CISA KEV** — risk‑informed triage (severity × exploit likelihood × active exploitation).
- **India** — DPDP privacy‑by‑design & consent flows; **CERT‑In** 6‑hour incident reporting in runbooks.

**Deliverables (keep in repo root):**
- `security/SECURITY-IRONCLAD.md` (this file)  
- `security/policy/ci-security-gates.yaml` (quality gates)  
- `security/agents/DevSecOps_gemini.md` (agent bootstrap) + root `gemini.md` that mirrors it  
- Control matrix template: `controls/matrix-asvs5-ssdf-slsa.xlsx` (optional)  
- Runbooks: `runbooks/zero-day.md`, `runbooks/incident-6hr-certin.md` (optional)

---

## 1) Architecture: End‑to‑End Trust (Zero Trust for SDLC)

**1.1 Zero Trust for CI/CD**
- Federated workload identities (OIDC → cloud), short‑lived tokens, least‑privilege scopes.  
- Segmented runners/builders; no shared workers; secrets via vault; no long‑lived PATs.  
- Admission control: image signature verification, SLSA provenance checks, policy‑as‑code.  
- Runtime baselines via **eBPF** for build agents & workloads; alert on deviations.

**1.2 Supply‑Chain Controls**
- Private proxy registries; **namespace allow‑listing** to defeat dependency confusion/typosquatting.  
- Mandatory **SBOM** (CycloneDX/SPDX) emitted at build; centralized SBOM store + **VEX** ingestion.  
- Pin hashes/lockfiles; renovate bots with policy; signed artifacts (cosign); provable reproducibility where feasible.

**1.3 Crypto‑Agility & PQC**
- Crypto inventory (algorithms/keys/protocols) with deprecation plan + automatic key rotation.  
- Pilot PQC hybrid (e.g., TLS 1.3 + ML‑KEM/Kyber hybrid) in non‑customer paths; maintain fallbacks.  
- KMS‑enforced lifetimes; HSM where required; no custom crypto.

---

## 2) Methodological Arsenal

**SAST/DAST/IAST/SCA** — IDE + CI (SAST/SCA), staging (DAST/IAST), periodic full scans; quarterly rule tuning.  
**IaC/Cloud** — scan Terraform/CloudFormation/K8s; enforce admission policies (non‑root, read‑only FS, capabilities drop, seccomp/apparmor).  
**Runtime** — eBPF (Falco/Tetragon) for syscall anomalies, K8s drift, exec‑in‑container.  
**LLM/AI** — apply OWASP LLM Top 10; red‑team for prompt injection, data exfiltration, jailbreaks, model inversion, membership inference; sign model artifacts; reproducible training; dataset integrity.  
**BAS** — continuous breach & attack simulation (e.g., CALDERA/Atomic Red Team) to validate detective/preventive controls.

---

## 3) Governance, People, Process

- **Policies**: Secure coding, Secrets mgmt, AI/ML model security, PQC roadmap, Offboarding/Insider threat (auto‑revoke VCS/build/cloud; rotate org tokens).  
- **ASPM**: unify SAST/SCA/DAST/IaC/Runtime into risk graph; dedupe; EPSS/KEV enrichment; dev self‑service dashboards.  
- **Privacy‑by‑Design (DPDP)**: PIA templates, data minimization, consent logs, retention & deletion SLAs.  
- **Security Champions**: 1+ per team; quarterly deep‑dives, gamified CTFs on real findings.

---

## 4) Vulnerability Management (Risk‑Based)

**Triage** = CVSS v4.0 × EPSS × KEV × Asset criticality × Exposure (internet?) × Compensating controls.  
**SLAs** (reference): Critical (KEV/actively exploited): **72h** remediate/mitigate (WAF/flags/config lock); High: **7d**; Medium: **30d**; Low: **90d**.  
**Zero‑Day**: immediate mitigations + 24h exec comms → weekly retro → purple‑team test; evidence logged.  
**Tracking**: single ticket queue; owner/due/rollback tested; auto re‑scan to verify.

---

## 5) Metrics & Scorecards

Blend **DORA** (deployment freq, lead time) with security SLAs:  
- MTTD/MTTR (critical)  
- % builds blocked by gates (should trend ↓ as tuning improves)  
- Coverage: ≥95% repos SAST/SCA; ≥90% services DAST/IaC; ≥80% runtime with eBPF sensors  
- % KEV vulns closed within SLA; recurrence rate; SBOM freshness ≤ 30 days  
- AI/LLM findings burn‑down; % signed images enforced; PQC pilot coverage

Dashboards by Product/Tribe/Org; monthly governance review with executive heatmaps.

---

## 6) Roadmap (Opinionated)

**0–90 days**  
- Pilot **ASVS 5.0** on one Tier‑0 service; publish mappings.  
- Enable org‑wide secret scanning & push protection; baseline SAST/SCA across repos.  
- Enforce image signature verify at K8s admission; stand up SBOM registry; wire **EPSS** into triage.

**3–6 months**  
- Admission policies: rootless, read‑only FS, capability drop; eBPF sensors + 10 high‑signal rules.  
- BAS pilot for key attack paths; remediate gaps.  
- AI/LLM red‑team v1; Champions trained on LLM Top 10.

**6–12 months**  
- **SLSA v1.1** provenance on all release artifacts; verify in CD.  
- Org‑wide ASVS 5.0 coverage; ZTA review complete.  
- PQC hybrid pilot results + crypto deprecation plan; migration runbook.  
- ASPM rollout with dev self‑service; auto‑dedupe & risk correlation.

---

## 7) Checklists (Extracts)

**Kubernetes Runtime (min):**  
- eBPF sensor deployed in all clusters  
- Admit‑time: verify signature, non‑root, capabilities drop, read‑only FS, seccomp/apparmor  
- NetworkPolicies default‑deny; secrets via KMS; rotate service‑account tokens

**Insider/Offboarding:**  
- Revoke VCS, package registry, CI, cloud, vault; rotate org tokens; audit last 30d; freeze suspicious pipelines; archive access logs

**Dependency Hygiene:**  
- Allow‑listed namespaces only; private proxy; quarantine unknowns  
- Pin SHAs; lockfiles in repo; automated update bot with policy

**Zero‑Day:**  
- 24h mitigations + comms; 72h patch/compensating control; retro + purple‑team test

---

## Appendix A — Sample Security Gates (YAML)

See `security/policy/ci-security-gates.yaml` in this baseline for a ready‑to‑use reference.

---

## Curated References
OWASP **ASVS 5.0**, NIST **SSDF v1.1** (SP 800‑218) + 218A guidance, **SLSA v1.1**, **CVSS v4.0**, **EPSS**, **CISA KEV**, NIST **SP 800‑207** (ZTA), India **DPDP Act 2023**, **CERT‑In** 6‑hour directions, GitHub Push Protection, npm/yarn namespace policies, Kyverno `verifyImages`, Falco/Tetragon eBPF runtime, OWASP **LLM Top 10**, MITRE ATT&CK, CALDERA, Atomic Red Team, OQS PQC.

> **Placement:** Keep this file under `security/` in every repo. Treat it like code—pull requests, reviews, and semantic versioning.

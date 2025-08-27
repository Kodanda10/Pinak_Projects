# Security Audit Guide

This guide describes how to perform a lightweight security audit of Pinak.

Checks
- Baseline files present: `security/SECURITY-IRONCLAD.md`, `security/policy/ci-security-gates.yaml`, `SECURITY.md`, `.well-known/security.txt`.
- Bridge secrets not tracked: `.pinak/` must not be tracked by git (`pinak doctor` reports if it is).
- CI security gates pass: gitleaks (secrets), semgrep (SAST), pip-audit (informational).
- JWT usage: tokens include `pid` and are verified with `SECRET_KEY`.
- Gateway isolation: `X-Pinak-Project` is enforced, and `pid` must match.
- RBAC: `PINAK_ALLOWED_ROLES` reflects organizational policy; tokens include role where appropriate.

Runbook
1. Run `pinak doctor` and address any issues.
2. Ensure CI security workflow is green on main and for PRs.
3. Review semgrep SARIF in GitHub Security tab (RESULTS) for trends and new findings.
4. Verify Docker images are pinned (`PARLANT_IMAGE_REF` supports digest pinning) and built via CI.
5. Optional: Enable pre-commit.ci to enforce hooks on PRs.

Notes
- Suppress semgrep findings only with strong rationale (`# nosemgrep`) and file an issue to remediate.
- For production, rotate `SECRET_KEY` and use a KMS or secrets manager; avoid hardcoding.

# CI![CI Tests](https://github.com/Pinak-Setu/Pinak_Projects/actions/workflows/ci-tests.yml/badge.svg)
![CI Security](https://github.com/Pinak-Setu/Pinak_Projects/actions/workflows/ci-security.yml/badge.svg)

**✅ CURRENT STATUS: All CI/CD jobs passing consistently**tus

Badges (replace owner/repo if different):

![CI Tests](https://github.com/Pinaka10/Pinak_Projects/actions/workflows/ci-tests.yml/badge.svg)
![CI Security](https://github.com/Pinaka10/Pinak_Projects/actions/workflows/ci-security.yml/badge.svg)

On failures, PRs receive an automated comment with a link to the failing run.

## CI Workflows

### CI Tests Workflow - ✅ FULLY OPERATIONAL

**Jobs Overview:**
- **vendor-tests**: Memory service tests with mock embeddings (2m58s avg)
- **gateway-tests**: RBAC and security tests (18s avg)
- **package-tests**: Integration tests with mock server (37s avg)
- **cli-smoke**: End-to-end CLI functionality tests (32s avg)

**Recent Critical Fixes (2025-08-30):**
- **✅ RESOLVED:** Cross-platform dependency conflicts (`pyobjc`, `rumps`)
- **✅ IMPLEMENTED:** Platform-specific dependency management
- **✅ ADDED:** Missing test dependencies (`requests` library)
- **✅ ENHANCED:** Mock server startup reliability and debugging

### Platform-Specific Dependencies

**macOS Dependencies (Local Development):**
- `pyobjc` - macOS Objective-C bridge for system integration
- `rumps` - macOS menubar framework (depends on pyobjc)

**Cross-Platform Dependencies (CI/CD Compatible):**
- All other dependencies work on both macOS and Ubuntu
- CI/CD environment correctly excludes macOS-specific packages
- Local macOS development preserves full functionality

### Security gate tuning

- Semgrep blocks ERROR-level findings; WARNINGs are informational. Prefer fixing findings; where noise is unavoidable, suppress with inline `nosemgrep` and rationale comment.
- gitleaks blocks hard failures; add false-positive patterns only with justification.
- Dependency audit runs informationally to avoid flakiness; pin versions to remediate high-risk CVEs.

## Pre-commit

We include a `.pre-commit-config.yaml`. To enable locally:

```
pip install pre-commit
pre-commit install
```

Optionally enable pre-commit.ci for this repository to enforce hooks on PRs. See https://pre-commit.ci/ for onboarding.

## Container Images

- Memory API: Published to GHCR by default; Docker Hub optional when secrets are set.
- Parlant: Build pipeline publishes to GHCR (and optionally Docker Hub). Use `PARLANT_IMAGE_REF` to pin digest in compose.

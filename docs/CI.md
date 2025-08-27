# CI Status

Badges (replace owner/repo if different):

![CI Tests](https://github.com/Pinaka10/Pinak_Projects/actions/workflows/ci-tests.yml/badge.svg)
![CI Security](https://github.com/Pinaka10/Pinak_Projects/actions/workflows/ci-security.yml/badge.svg)

On failures, PRs receive an automated comment with a link to the failing run.

## CI Workflows

- CI Security Gates: gitleaks + semgrep (blocking), pip-audit informational.
- CI Tests: runs memory service tests with mock embeddings.
- Build Parlant: builds Parlant from vendored pip-build Dockerfile and publishes to GHCR (and Docker Hub if secrets set).

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

# Contributing

## Development Environment Setup

### Prerequisites
- **Python 3.13+** (macOS development)
- **Python 3.11+** (CI/CD compatibility)
- **Git** with GitHub access
- **Redis** (optional, for caching features)

### Platform-Specific Setup

**macOS Development (Full Feature Set):**
```bash
# Install all dependencies including macOS-specific packages
pip install -e .
# This will install: pyobjc, rumps, and all other dependencies
```

**Cross-Platform Development (Core Features):**
```bash
# Install core dependencies (CI/CD compatible)
pip install -r requirements.txt
pip install -e .
# macOS-specific packages (pyobjc, rumps) will be skipped automatically
```

## Pre-commit

Install pre-commit and enable hooks to catch issues early:

```
pip install pre-commit
pre-commit install
```

This runs hygiene checks (EOL, whitespace), gitleaks (secrets), and semgrep (SAST) before commits.

## Security Gates

CI enforces gates per `security/policy/ci-security-gates.yaml`. Keep PRs green by:
- Fixing any semgrep findings or suppressing with rationale.
- Avoiding secrets in code; let gitleaks pass.
- Running `pinak doctor` to ensure baseline files exist.

## Tests

### Local Testing
Run vendored service tests:

```
cd Pinak_Services/memory_service
USE_MOCK_EMBEDDINGS=true pytest -q
```

### CI/CD Testing
All tests run automatically on:
- **vendor-tests**: Memory service with mock embeddings
- **gateway-tests**: RBAC and security tests
- **package-tests**: Integration tests with mock server
- **cli-smoke**: End-to-end CLI functionality tests

**✅ Current Status:** All 4 CI/CD jobs passing consistently

## One-click

Use `pinak quickstart` for local setup. Set registry env when using published images:

```
export PINAK_IMAGE_REGISTRY=PINAK10
pinak quickstart --name "MyApp" --url http://localhost:8011
```

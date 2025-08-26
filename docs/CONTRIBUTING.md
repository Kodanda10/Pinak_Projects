# Contributing

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

Run vendored service tests:

```
cd Pinak_Services/memory_service
USE_MOCK_EMBEDDINGS=true pytest -q
```

## One-click

Use `pinak quickstart` for local setup. Set registry env when using published images:

```
export PINAK_IMAGE_REGISTRY=PINAK10
pinak quickstart --name "MyApp" --url http://localhost:8011
```


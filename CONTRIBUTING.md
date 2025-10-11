# Contributing to Pinak

Thank you for your interest in contributing to Pinak! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Testing Requirements](#testing-requirements)
- [Code Style](#code-style)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Security](#security)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and professional in all interactions.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

### Prerequisites

- Python 3.11 or higher
- [uv](https://astral.sh/uv) package manager
- Git
- Docker (optional, for containerized development)

### Setting Up Development Environment

1. **Fork and Clone**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Pinak_Projects.git
   cd Pinak_Projects
   ```

2. **Install uv**
   ```bash
   pip install --upgrade uv
   ```

3. **Sync Dependencies**
   ```bash
   uv sync --frozen
   uv sync --project Pinak_Services/memory_service --extra tests --frozen
   ```

4. **Configure Environment**
   ```bash
   export PINAK_JWT_SECRET="dev-secret-change-me"
   export PINAK_EMBEDDING_BACKEND="dummy"  # For testing
   ```

5. **Run Tests**
   ```bash
   cd Pinak_Services/memory_service
   uv run pytest tests/ -v
   ```

## Development Workflow

### Test-Driven Development (TDD)

We strictly follow TDD principles:

1. **Write Failing Tests First**
   - Write unit tests that fail initially
   - Tests should cover the desired functionality
   - Run tests to confirm they fail (`pytest tests/ -v`)

2. **Implement Minimal Code**
   - Write the minimum code to make tests pass
   - Focus on functionality, not perfection
   - Run tests to confirm they pass

3. **Refactor**
   - Clean up code while keeping tests green
   - Improve structure and readability
   - Ensure tests still pass after refactoring

### Branch Strategy

- `main` - Stable production-ready code
- `develop` - Integration branch for features
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates

### Creating a Feature

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Write failing tests
# Implement feature
# Ensure all tests pass

# Commit changes
git add .
git commit -m "feat: description of feature"

# Push and create PR
git push origin feature/your-feature-name
```

## Testing Requirements

### Test Coverage

- **Minimum**: 80% code coverage required
- **Goal**: 90%+ coverage for critical paths
- Run coverage: `pytest tests/ --cov=app --cov-report=html`

### Test Types

1. **Unit Tests**
   - Test individual functions/methods
   - Mock external dependencies
   - Fast execution (< 1s per test)

2. **Integration Tests**
   - Test component interactions
   - Use real dependencies where practical
   - Moderate execution time (< 5s per test)

3. **End-to-End Tests**
   - Test complete user workflows
   - Use test fixtures and temporary directories
   - May take longer (< 30s per test)

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_authorization.py -v

# Specific test
pytest tests/test_authorization.py::test_admin_has_all_permissions -v

# With coverage
pytest tests/ --cov=app --cov-report=term-missing

# Fast tests only (exclude slow markers)
pytest tests/ -v -m "not slow"
```

## Code Style

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line Length**: 120 characters (not 79)
- **Quotes**: Double quotes for strings
- **Imports**: Organized using `isort`
- **Formatting**: Use `black` for consistent formatting

### Linting

```bash
# Run flake8
flake8 Pinak_Services/memory_service --max-line-length=127

# Run black (check only)
black --check Pinak_Services/memory_service

# Run black (format)
black Pinak_Services/memory_service
```

### Type Hints

- Use type hints for all function signatures
- Use `typing` module for complex types
- Use `Optional` for nullable values

```python
from typing import List, Optional, Dict, Any

def process_data(items: List[str], config: Optional[Dict[str, Any]] = None) -> bool:
    """Process data with optional configuration."""
    pass
```

### Documentation

- **Docstrings**: Required for all public functions, classes, and modules
- **Format**: Google-style docstrings
- **Comments**: Explain *why*, not *what*

```python
def calculate_score(factors: List[int]) -> float:
    """Calculate weighted score from multiple factors.
    
    Args:
        factors: List of integer factor values
        
    Returns:
        Weighted score as float between 0.0 and 1.0
        
    Raises:
        ValueError: If factors list is empty
    """
    if not factors:
        raise ValueError("Factors list cannot be empty")
    return sum(factors) / len(factors) / 10.0
```

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

### Examples

```
feat(auth): add role-based authorization module

Implement RBAC with four roles (admin, user, guest, service) and 
granular permissions. Includes comprehensive unit and integration tests.

Closes #123
```

```
fix(memory): correct timestamp handling in event log

Replace deprecated datetime.utcnow() with timezone-aware 
datetime.now(datetime.UTC) to fix deprecation warnings.
```

## Pull Request Process

### Before Submitting

1. **Update from main**
   ```bash
   git checkout main
   git pull origin main
   git checkout your-branch
   git rebase main
   ```

2. **Run all tests**
   ```bash
   pytest tests/ -v
   ```

3. **Check linting**
   ```bash
   flake8 Pinak_Services/memory_service
   ```

4. **Update documentation**
   - Update relevant README sections
   - Add/update docstrings
   - Update CHANGELOG.md if applicable

### PR Requirements

- [ ] All tests pass
- [ ] Code coverage ≥ 80%
- [ ] Linting checks pass
- [ ] Documentation updated
- [ ] Commit messages follow guidelines
- [ ] No merge conflicts
- [ ] Related issue linked (if applicable)

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Checklist
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] No linting errors
- [ ] Follows TDD principles

## Related Issues
Closes #XXX
```

### Review Process

1. **Automated Checks**: CI/CD must pass
2. **Code Review**: At least one approval required
3. **Testing**: Reviewer verifies test coverage
4. **Documentation**: Reviewer checks docs
5. **Merge**: Squash and merge to main

## Security

### Reporting Security Issues

**DO NOT** open public issues for security vulnerabilities.

Instead:
- Email: security@pinak-setu.com
- See [SECURITY.md](SECURITY.md) for details

### Security Checklist

- [ ] No hardcoded secrets or credentials
- [ ] Sensitive data properly redacted in logs
- [ ] Input validation on all user inputs
- [ ] Authentication/authorization properly enforced
- [ ] Dependencies scanned for vulnerabilities

## Additional Resources

- [README](README.md) - Project overview
- [SECURITY.md](SECURITY.md) - Security policies
- [API Documentation](docs/)
- [Architecture Guide](pinak_enterprise_reference.md)

## Questions?

- **Issues**: [GitHub Issues](https://github.com/Pinak-Setu/Pinak_Projects/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Pinak-Setu/Pinak_Projects/discussions)
- **Email**: support@pinak-setu.com

---

**Thank you for contributing to Pinak!** 🏹

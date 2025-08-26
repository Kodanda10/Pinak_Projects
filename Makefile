.PHONY: test audit up down build

test:
	python -m pip install --upgrade pip >/dev/null || true
	pip install -e . >/dev/null
	pip install pytest respx >/dev/null
	pytest -q

audit:
	@echo "Run security audits (gitleaks/semgrep/checkov/pip-audit/safety) via CI" 

up:
	$(MAKE) -C Pinak_Services up

down:
	$(MAKE) -C Pinak_Services down

build:
	python -m build || true


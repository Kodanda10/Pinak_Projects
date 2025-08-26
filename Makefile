.PHONY: test audit up down build bridge-init bridge-status bridge-verify dev-token ensure-venv

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

ensure-venv:
	@if [ ! -d .venv ]; then python3 -m venv .venv; fi
	.venv/bin/python -m pip install --upgrade pip >/dev/null
	.venv/bin/pip install -e . >/dev/null

# Initialize Pinak Bridge for this repo
bridge-init: ensure-venv
	@.venv/bin/pinak-bridge init --name "$(name)" --url "$(url)" --tenant "$(tenant)" $(if $(token),--token "$(token)")

# Show current bridge status
bridge-status: ensure-venv
	@.venv/bin/pinak-bridge status

# Verify .pinak config, fingerprint, and gitignore guards
bridge-verify: ensure-venv
	@.venv/bin/pinak-bridge verify

# Mint a dev JWT and optionally set it into the bridge
dev-token: ensure-venv
	@./scripts/dev_token.sh $(if $(sub),--sub "$(sub)") $(if $(secret),--secret "$(secret)") $(if $(set),--set)

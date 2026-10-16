.PHONY: help setup dev-api dev-web lint typecheck test test-slow check build up down ps deploy undeploy status logs

SHELL := bash
.SHELLFLAGS := -euo pipefail -c

UNIT_DIR := $(HOME)/.config/systemd/user
UNIT     := framehound

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────────────────────────────────

setup: ## Install the Python and web packages
	uv sync --all-packages --all-extras
	cd web && pnpm install --frozen-lockfile

# ── Dev ───────────────────────────────────────────────────────────────────────

dev-api: ## Run the api on :8000 with the tiny models, local indexing, jobs in threads
	DATA_DIR=./data FH_MODELS_DIR=./data/models FH_MODELS_LOCK=models.tiny.lock \
		INDEX_BACKEND=local uv run framehound-api

dev-web: ## Run the Vite dev server on :5173 (proxies /api to :8000)
	cd web && pnpm dev

# ── Checks ────────────────────────────────────────────────────────────────────

lint: ## Ruff
	uv run ruff check server indexer

typecheck: ## TypeScript
	cd web && pnpm typecheck

test: ## Tests (Colab is faked)
	uv run pytest

test-slow: ## The real indexing pipeline with the tiny models
	FH_SLOW_TESTS=1 uv run pytest indexer

check: lint typecheck test ## Lint, typecheck, and test

# ── Stack ─────────────────────────────────────────────────────────────────────

build: ## Build the docker images
	docker compose build

up: ## Start the stack (http://localhost:8080)
	docker compose up -d --wait

down: ## Stop the stack
	docker compose down

ps: ## Show the stack's containers
	docker compose ps

# ── Deploy ────────────────────────────────────────────────────────────────────

deploy: build ## Build, install the systemd user unit, and (re)start it on the tailnet
	install -Dm644 docker/$(UNIT).service $(UNIT_DIR)/$(UNIT).service
	systemctl --user daemon-reload
	systemctl --user enable $(UNIT)
	systemctl --user restart $(UNIT)

undeploy: ## Stop and remove the systemd user unit and its tailnet mapping
	systemctl --user disable --now $(UNIT) || true
	rm -f $(UNIT_DIR)/$(UNIT).service
	systemctl --user daemon-reload

status: ## Show the unit and the tailnet mapping
	systemctl --user --no-pager status $(UNIT) || true
	tailscale serve status

logs: ## Follow the stack's logs
	docker compose logs -f

.PHONY: help up down logs ingest eval mcp-server test lint qa profile

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up: ## Build and start Ollama + app via Docker Compose
	docker compose up -d --build

down: ## Stop all services
	docker compose down

logs: ## Tail the app logs
	docker compose logs -f app

ingest: ## Ingest the sample PDF into the app container
	docker compose exec app python src/ingestion/ingest.py

eval: ## Run the hybrid retrieval benchmark inside the app container
	docker compose exec app python src/evaluation/run_eval.py --k 5 --hybrid

eval-rag: ## Run the full RAG benchmark (LLM-as-a-judge) inside the app container
	docker compose exec app python src/evaluation/run_eval.py --k 5 --hybrid --rag

mcp-server: ## Run the MCP server on stdio (for Claude Desktop / other MCP clients)
	python src/mcp_server.py

test: ## Run the offline test suite
	python -m pytest tests/ -q

lint: ## Lint the codebase with pylint and ruff
	pylint $(shell git ls-files '*.py')
	ruff check .
	ruff format --check .

qa: ## Full QA gate: lint, types, security, tests with coverage
	$(MAKE) lint
	pyright .
	bandit -r src/
	python -m coverage run --source=src -m pytest tests/ -q
	python -m coverage report -m

profile: ## Profile a representative offline workload with scalene
	scalene run --cpu-only -o /tmp/scalene-profile.json scripts/bench.py
	scalene view --cli /tmp/scalene-profile.json

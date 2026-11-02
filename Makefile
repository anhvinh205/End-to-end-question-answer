.DEFAULT_GOAL := help
PYTHON ?= python
CONFIG ?= configs/config.yaml

.PHONY: help install data dev-official index train evaluate evaluate-bm25 evaluate-official api ui test test-cov lint format typecheck check docker-up clean

help: ## List the available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with all extras (editable)
	$(PYTHON) -m pip install -e ".[all]"

data: ## Clean the Natural Questions CSV, split by passage, write the audit report
	$(PYTHON) scripts/prepare_data.py --config $(CONFIG)

dev-official: ## Stream and prepare the official 7,830 NQ dev set from Hugging Face
	$(PYTHON) scripts/prepare_official_dev.py --config $(CONFIG)

index: ## Build the FAISS + BM25 knowledge-base index
	$(PYTHON) scripts/build_index.py --config $(CONFIG)

train: ## Fine-tune the reader on NQ and tune its abstention threshold (GPU recommended)
	$(PYTHON) scripts/train_reader.py --config $(CONFIG)

evaluate: ## Retrieval + reader + end-to-end metrics -> reports/metrics.{json,md}
	$(PYTHON) scripts/evaluate.py --config $(CONFIG)

evaluate-bm25: ## Retrieval baseline only (needs data, no model download)
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --bm25-only

evaluate-official: ## Evaluate retrieval + reader + end-to-end on the official dev set
	$(PYTHON) scripts/evaluate.py --config $(CONFIG) --official-dev

api: ## Run the REST API on :8000 (falls back to the bundled demo corpus)
	uvicorn nq_qa.api.main:app --host 0.0.0.0 --port 8000 --reload

ui: ## Run the Streamlit demo on :8501
	streamlit run app/streamlit_app.py

test: ## Run the test suite (offline, tiny models)
	$(PYTHON) -m pytest -q

test-cov: ## Tests with a coverage report and a 90 % floor
	$(PYTHON) -m pytest -q --cov=nq_qa --cov-report=term-missing --cov-fail-under=90

lint: ## Ruff lint + format check
	ruff check .
	ruff format --check .

format: ## Auto-fix lint issues and format
	ruff check --fix .
	ruff format .

typecheck: ## mypy
	mypy

check: lint typecheck test ## Everything CI runs

docker-up: ## Build and start API + UI with docker compose
	docker compose up --build

clean: ## Remove caches and build outputs
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage coverage.xml build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

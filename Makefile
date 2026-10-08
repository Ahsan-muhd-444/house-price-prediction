# Development tasks. Run `make help` for the list.
PYTHON ?= python

.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test build docs docs-serve docker benchmark clean

help: ## Show this help
	@$(PYTHON) -c "import re; [print(f'  {m[0]:<12} {m[1]}') for m in re.findall(r'^([a-z-]+):.*?## (.*)$$', open('Makefile').read(), re.M)]"

install: ## Install the package with every development extra
	$(PYTHON) -m pip install -e ".[dev,docs]"
	pre-commit install

lint: ## Lint and check formatting
	ruff check .
	ruff format --check .

format: ## Apply fixes and formatting
	ruff check --fix .
	ruff format .

typecheck: ## Static type checking
	mypy src

test: ## Run the test suite with coverage
	$(PYTHON) -m pytest

build: ## Build the sdist and wheel, then check them
	$(PYTHON) -m build
	twine check dist/*

docs: ## Build the documentation site, failing on warnings
	mkdocs build --strict

docs-serve: ## Preview the documentation at http://127.0.0.1:8000
	mkdocs serve

docker: ## Build the API image
	docker build -t amesml:local .

benchmark: ## Re-run the benchmarks and rewrite docs/benchmarks.md
	$(PYTHON) scripts/benchmark.py

clean: ## Remove build artefacts and caches
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in ('build', 'dist', 'site', 'htmlcov', '.mypy_cache', '.ruff_cache', '.pytest_cache')]; [p.unlink() for p in pathlib.Path('.').glob('.coverage*')]"

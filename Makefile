.PHONY: help install install-dev test test-unit test-integration test-coverage lint format type-check clean demo serve docs

help: ## Show this help message
	@echo "AUREON - AI/ML Pipeline Management System"
	@echo "=========================================="
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install AUREON in production mode
	pip install -e .

install-dev: ## Install AUREON in development mode with all dependencies
	pip install -e ".[dev,docs,jupyter]"
	pre-commit install

test: ## Run all tests
	pytest tests/ -v

test-unit: ## Run unit tests only
	pytest tests/ -m unit -v

test-integration: ## Run integration tests only
	pytest tests/ -m integration -v

test-coverage: ## Run tests with coverage report
	pytest tests/ --cov=aureon --cov-report=html --cov-report=term

lint: ## Run linting checks
	flake8 aureon/ tests/ scripts/
	black --check aureon/ tests/ scripts/

format: ## Format code with black
	black aureon/ tests/ scripts/

type-check: ## Run type checking with mypy
	mypy aureon/

clean: ## Clean up generated files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

demo: ## Run the demo script
	python scripts/demo.py

serve: ## Start the API server
	aureon serve

generate-data: ## Generate sample datasets
	python scripts/generate_sample_data.py

docs: ## Generate documentation
	@echo "Documentation generation not implemented yet"

check: lint type-check test ## Run all quality checks

ci: clean install-dev check ## Run CI pipeline locally

build: clean ## Build the package
	python setup.py sdist bdist_wheel

upload: build ## Upload to PyPI (requires credentials)
	twine upload dist/*

docker-build: ## Build Docker image
	docker build -t aureon:latest .

docker-run: ## Run Docker container
	docker run -p 8000:8000 aureon:latest

quick-start: install generate-data demo ## Quick start guide

all: clean install-dev check demo ## Run everything

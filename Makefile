.PHONY: help env install test test-nb lint clean

PYTHON := $(shell which python)
CONDA_ENV := cheminf

help:
	@echo "Cheminformatics Course Repository"
	@echo "  make env       - Create conda environment from environment.yml"
	@echo "  make install   - Install chemlib in editable mode"
	@echo "  make test      - Run unit tests with pytest"
	@echo "  make test-nb   - Run smoke tests on notebooks"
	@echo "  make lint      - Lint code with ruff"
	@echo "  make clean     - Strip notebook outputs and remove caches"

install:
	pip install -e .

test:
	pytest tests/

test-nb:
	python -m pytest --nbval-lax notebooks/

lint:
	ruff check src/ tests/

clean:
	nbstripout notebooks/*.ipynb solutions/*.ipynb assignments/*/*.ipynb 2>/dev/null || true
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

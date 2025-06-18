.PHONY: help install install-dev test lint format type-check clean build upload docs

help:
	@echo "Available commands:"
	@echo "  install      Install the package"
	@echo "  install-dev  Install development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code with black"
	@echo "  type-check   Run type checking with mypy"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build the package"
	@echo "  upload       Upload to PyPI"
	@echo "  docs         Generate documentation"

install:
	pip install .

install-dev:
	pip install -e ".[dev]"
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v --cov=src/pega_tools --cov-report=html --cov-report=term

lint:
	flake8 src/ tests/

format:
	black src/ tests/

type-check:
	mypy src/pega_tools/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

upload: build
	twine upload dist/*

docs:
	cd docs && make html 
.PHONY: install lint test figures

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

figures:
	uv run --group report python scripts/make_figures.py

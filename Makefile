.PHONY: install lint test figures bench

install:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

figures:
	uv run --group report python scripts/make_figures.py

bench:
	uv run python evals/run_benchmark.py
	uv run python evals/aggregate.py

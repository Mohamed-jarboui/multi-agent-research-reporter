.PHONY: install lint test figures bench serve docker

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

serve:
	uv run uvicorn reporter.service.app:app --host 127.0.0.1 --port 7860

docker:
	docker build -t research-reporter .

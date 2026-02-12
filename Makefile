.PHONY: all lint format typecheck test test-cov build docs \
       docker-build docker-up docker-down clean

all: lint typecheck test

# ---------- Code quality ----------

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run pyright

# ---------- Testing ----------

test:
	uv run pytest -v --tb=short

test-cov:
	uv run pytest --cov=memfun --cov-report=html

# ---------- Build ----------

build:
	uv build

docs:
	mkdocs build

# ---------- Docker ----------

docker-build:
	docker build -f docker/Dockerfile -t memfun .

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down

# ---------- Cleanup ----------

clean:
	rm -rf dist .pytest_cache .ruff_cache __pycache__

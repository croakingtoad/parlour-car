.PHONY: dev dev-down test lint format typecheck run worker clean

# Start development infrastructure (PostgreSQL + Neo4j)
dev:
	sg docker -c "docker compose up -d"

# Stop development infrastructure
dev-down:
	sg docker -c "docker compose down"

# Run tests
test:
	uv run pytest tests/ -v

# Run tests with coverage
test-cov:
	uv run pytest tests/ -v --cov=author_library --cov-report=term-missing

# Lint source code
lint:
	uv run ruff check src/ tests/

# Format source code
format:
	uv run ruff format src/ tests/

# Run type checking
typecheck:
	uv run mypy src/

# Start the MCP server
run:
	uv run python -m author_library

# Start the arq background worker
worker:
	uv run arq author_library.worker.WorkerSettings

# Remove build artifacts
clean:
	rm -rf .mypy_cache .pytest_cache .ruff_cache dist/ build/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

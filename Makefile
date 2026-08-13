.PHONY: dev dev-down test test-cov test-db-up test-db-down test-db-reset lint format typecheck run worker clean

# Bolt URL of the disposable test Neo4j (compose service neo4j-test).
# Tests refuse to run against a graph holding production data — see
# tests/conftest.py::assert_graph_is_disposable.
TEST_NEO4J_URL ?= bolt://localhost:7688

# Start development infrastructure (PostgreSQL + Neo4j)
dev:
	sg docker -c "docker compose up -d"

# Stop development infrastructure
dev-down:
	sg docker -c "docker compose down"

# Start the disposable test Neo4j (tmpfs-backed, port 7688)
test-db-up:
	sg docker -c "docker compose --profile test up -d neo4j-test"
	@echo "waiting for neo4j-test to become healthy..."
	@for i in $$(seq 1 40); do \
		status=$$(sg docker -c "docker inspect -f '{{.State.Health.Status}}' parlour-neo4j-test" 2>/dev/null); \
		[ "$$status" = "healthy" ] && echo "neo4j-test healthy on 7688" && exit 0; \
		sleep 2; \
	done; \
	echo "neo4j-test did not become healthy" >&2; exit 1

# Stop and discard the test Neo4j (tmpfs — all data goes with it)
test-db-down:
	sg docker -c "docker compose --profile test rm -sf neo4j-test"

# Recreate the test graph from scratch.
#
# Cleanup fixtures are prefix-scoped to test-- and must NEVER orphan-sweep,
# so entity nodes the LLM names itself (Theme "imagination-and-theology",
# Person, Concept, Argument) cannot be removed by prefix and survive teardown.
# Left in place they leak across suites — tests/test_dashboard asserts an empty
# graph reports no orphaned themes and fails if a previous suite left any.
# The graph is tmpfs-backed, so recreating it is the cheap, correct reset.
test-db-reset: test-db-down test-db-up

# Run tests against a pristine disposable graph, never the production one
test: test-db-reset
	TEST_NEO4J_URL=$(TEST_NEO4J_URL) uv run pytest tests/ -v

# Run tests with coverage
test-cov: test-db-reset
	TEST_NEO4J_URL=$(TEST_NEO4J_URL) uv run pytest tests/ -v \
		--cov=author_library --cov-report=term-missing

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

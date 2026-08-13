"""Root test configuration — ensures tests NEVER touch the production database.

Sets DB_POSTGRES_URL to a dedicated test database before any
DatabaseSettings instances are constructed. Every sub-conftest that
uses DatabaseSettings() will automatically receive the test DB URL.

Override defaults with environment variables:
    TEST_POSTGRES_URL  (default: author_library_test on localhost)
    TEST_NEO4J_URL     (default: bolt://localhost:7687)

PostgreSQL gets a genuinely separate database (author_library_test), but
Neo4j Community Edition serves a single database, so TEST_NEO4J_URL points
at the same graph the production corpus lives in. Prefix-scoped cleanup is
the only thing keeping test teardown off real data — and that has failed
twice (2026-07-02 theme wipe, 2026-08-13 Guite work wipe). The guard fixture
below refuses to run graph tests against a graph holding production data
unless the operator opts in explicitly.
"""

from __future__ import annotations

import os

import pytest

# ---------------------------------------------------------------------------
# CRITICAL: Override database URLs BEFORE any pydantic-settings model loads.
# DatabaseSettings uses env_prefix="DB_", so setting DB_POSTGRES_URL here
# ensures every DatabaseSettings() construction points at the test database.
# ---------------------------------------------------------------------------

_TEST_POSTGRES_URL = os.environ.get(
    "TEST_POSTGRES_URL",
    "postgresql://author_library:author_library@localhost:5432/author_library_test",
)
_TEST_NEO4J_URL = os.environ.get(
    "TEST_NEO4J_URL",
    "bolt://localhost:7687",
)

os.environ["DB_POSTGRES_URL"] = _TEST_POSTGRES_URL
os.environ["DB_NEO4J_URL"] = _TEST_NEO4J_URL


# ---------------------------------------------------------------------------
# Production-graph guard
# ---------------------------------------------------------------------------

#: Every work_id / canonical_name a test creates in Neo4j must start with this.
TEST_NAMESPACE = "test--"

#: Escape hatch. Set to "1" to allow graph tests against a graph that holds
#: production data. Only do this if you accept that a buggy teardown can
#: delete real, expensive data.
_ALLOW_PRODUCTION_GRAPH = os.environ.get("PARLOUR_ALLOW_PRODUCTION_GRAPH") == "1"


@pytest.fixture(scope="session")
def assert_graph_is_disposable() -> None:
    """Fail the test session if the target graph holds production data.

    Request this fixture from any conftest that hands out a Neo4j
    connection. It is deliberately not autouse: unit tests that never
    open a graph connection should not pay for a round trip.
    """
    if _ALLOW_PRODUCTION_GRAPH:
        return

    from neo4j import GraphDatabase

    user = os.environ.get("DB_NEO4J_USER", "neo4j")
    password = os.environ.get("DB_NEO4J_PASSWORD", "neo4j_dev")

    try:
        driver = GraphDatabase.driver(_TEST_NEO4J_URL, auth=(user, password))
        with driver.session() as session:
            record = session.run(
                "MATCH (w:Work) WHERE NOT w.work_id STARTS WITH $ns "
                "RETURN count(w) AS n, collect(w.work_id)[0..3] AS sample",
                ns=TEST_NAMESPACE,
            ).single()
        driver.close()
    except Exception as exc:
        pytest.skip(f"Neo4j not reachable at {_TEST_NEO4J_URL}: {exc}")

    if record and record["n"]:
        pytest.exit(
            f"\n\nREFUSING TO RUN GRAPH TESTS.\n"
            f"{_TEST_NEO4J_URL} holds {record['n']} production Work node(s), "
            f"e.g. {record['sample']}.\n"
            f"Test teardown deletes by work_id prefix and has destroyed real data "
            f"twice. Point TEST_NEO4J_URL at a disposable Neo4j instance, or set "
            f"PARLOUR_ALLOW_PRODUCTION_GRAPH=1 to override.\n",
            returncode=3,
        )

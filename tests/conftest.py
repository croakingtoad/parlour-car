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
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

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

#: Author slugs of the real corpus. A test must never create nodes under these:
#: doing so is indistinguishable from production data, which is what put a real
#: prefix into a cleanup fixture and deleted 5 works on 2026-08-13. Single
#: source of truth — tests/test_infrastructure/test_neo4j_cleanup_safety.py
#: enforces it statically and reset_disposable_graph checks it at runtime.
PRODUCTION_AUTHOR_PREFIXES = (
    "malcolm-guite",
    "samuel-taylor-coleridge",
    "george-macdonald",
    "henri-j-m-nouwen",
    "iain-mcgilchrist",
    "john-odonohue",
    "richard-holmes",
    "ewan-james-jones",
    "martin-shaw",
    "paul-david-tripp",
    "paul-kingsnorth",
)

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

    _refuse_if_production(record)


@pytest.fixture
def reset_disposable_graph() -> "Callable[[Any], Awaitable[None]]":
    """Clear the whole test graph, after re-proving it is disposable.

    This is the ONE audited place allowed to delete unscoped, and it earns that
    by re-running the production check immediately before deleting — so it
    cannot fire against a graph holding real works even if
    PARLOUR_ALLOW_PRODUCTION_GRAPH is set.

    Needed because cleanup is prefix-scoped to test--, which by design cannot
    remove entity nodes the LLM names itself (Theme "imagination-and-theology",
    Person, Concept, Argument). Those survive teardown and leak between suites,
    so a test asserting on an empty graph is otherwise order-dependent. Do not
    reintroduce an orphan sweep to solve that — see
    tests/test_infrastructure/test_neo4j_cleanup_safety.py.
    """

    async def _reset(neo4j: Any) -> None:
        # Checked against real author prefixes rather than "anything outside
        # test--": legacy fixtures still use ids like guite--/coleridge--/
        # lewis-- (see the namespace-unification task), so a strict test--
        # check would trip on the suite's own data mid-session. Production
        # work_ids always carry a full author slug, which fixtures never use.
        record = await neo4j.execute_read(
            "MATCH (w:Work) WHERE any(p IN $prefixes WHERE w.work_id STARTS WITH p) "
            "RETURN count(w) AS n, collect(w.work_id)[0..3] AS sample",
            {"prefixes": [f"{p}--" for p in PRODUCTION_AUTHOR_PREFIXES]},
        )
        row = record[0] if record else None
        if row and row["n"]:
            raise AssertionError(
                f"refusing to reset a graph holding {row['n']} production "
                f"Work node(s), e.g. {row['sample']}"
            )
        await neo4j.execute_write("MATCH (n) DETACH DELETE n", {})

    return _reset


def _refuse_if_production(record: Any) -> None:
    """Halt the session when the target graph holds production data."""
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

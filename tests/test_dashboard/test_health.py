"""Contract tests for dashboard health checks."""

from author_library.dashboard.health import (
    CheckResult,
    check_entity_extraction_gaps,
    check_low_confidence_classifications,
    check_missing_embeddings,
    check_orphaned_theme_nodes,
    check_pg_neo4j_sync,
    check_theme_coverage,
    check_unvoiced_primary_sources,
    run_all_checks,
)
from tests.test_dashboard.conftest import SKIP_NO_DB

VALID_STATUSES = {"ok", "warn", "error"}
ALL_CHECK_NAMES = {
    "pg_neo4j_sync",
    "missing_embeddings",
    "unvoiced_primary_sources",
    "low_confidence_classifications",
    "entity_extraction_gaps",
    "orphaned_theme_nodes",
    "theme_coverage",
}


@SKIP_NO_DB
class TestRunAllChecks:
    async def test_returns_seven_results(self, storage):
        results = await run_all_checks(storage.pg, storage.neo4j)
        assert len(results) == 7

    async def test_all_are_check_results(self, storage):
        for r in await run_all_checks(storage.pg, storage.neo4j):
            assert isinstance(r, CheckResult)

    async def test_all_statuses_valid(self, storage):
        for r in await run_all_checks(storage.pg, storage.neo4j):
            assert r.status in VALID_STATUSES, f"{r.name}: invalid status {r.status!r}"

    async def test_all_names_present(self, storage):
        names = {r.name for r in await run_all_checks(storage.pg, storage.neo4j)}
        assert names == ALL_CHECK_NAMES

    async def test_empty_db_no_crashes(self, clean_storage):
        results = await run_all_checks(clean_storage.pg, clean_storage.neo4j)
        for r in results:
            assert r.count != -1, f"{r.name} raised an exception on empty DB: {r.detail}"


@SKIP_NO_DB
class TestIndividualChecks:
    async def test_missing_embeddings_ok_on_empty_db(self, clean_storage):
        r = await check_missing_embeddings(clean_storage.pg)
        assert r.status == "ok"
        assert r.count == 0

    async def test_pg_neo4j_sync_ok_on_empty_db(self, clean_storage):
        r = await check_pg_neo4j_sync(clean_storage.pg, clean_storage.neo4j)
        assert r.status == "ok"

    async def test_low_confidence_ok_on_empty_db(self, clean_storage):
        r = await check_low_confidence_classifications(clean_storage.pg)
        assert r.status == "ok"
        assert r.count == 0

    async def test_orphaned_themes_ok_on_empty_neo4j(self, clean_storage):
        r = await check_orphaned_theme_nodes(clean_storage.neo4j)
        assert r.status == "ok"

    async def test_theme_coverage_ok_when_no_primary_works(self, clean_storage):
        r = await check_theme_coverage(clean_storage.pg, clean_storage.neo4j)
        assert r.status == "ok"
        assert "No primary works" in r.detail

    async def test_entity_gaps_ok_when_no_primary_works(self, clean_storage):
        r = await check_entity_extraction_gaps(clean_storage.pg, clean_storage.neo4j)
        assert r.status == "ok"

    async def test_unvoiced_ok_when_no_primary_works(self, clean_storage):
        r = await check_unvoiced_primary_sources(clean_storage.pg)
        assert r.status == "ok"

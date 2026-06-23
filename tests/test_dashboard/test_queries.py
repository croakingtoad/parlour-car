"""Shape tests for dashboard queries.

These run against author_library_test (the live test DB).
They verify the return structure, not specific counts,
since the test DB may be empty.
"""

from author_library.dashboard.queries import (
    get_graph_stats,
    get_library_overview,
    get_per_work_details,
)
from tests.test_dashboard.conftest import SKIP_NO_DB


@SKIP_NO_DB
class TestGetLibraryOverview:
    async def test_returns_required_keys(self, storage):
        result = await get_library_overview(storage.pg)
        assert isinstance(result, dict)
        for key in (
            "total_works", "primary_works", "secondary_works",
            "contextual_works", "tertiary_works", "personal_works",
            "total_chunks", "embedding_coverage_pct", "voice_profile_count",
        ):
            assert key in result, f"Missing key: {key}"

    async def test_counts_non_negative(self, storage):
        result = await get_library_overview(storage.pg)
        assert result["total_works"] >= 0
        assert result["total_chunks"] >= 0
        assert 0.0 <= result["embedding_coverage_pct"] <= 100.0

    async def test_coverage_does_not_crash_on_empty_db(self, clean_storage):
        result = await get_library_overview(clean_storage.pg)
        assert isinstance(result["embedding_coverage_pct"], float)


@SKIP_NO_DB
class TestGetPerWorkDetails:
    async def test_returns_list(self, storage):
        result = await get_per_work_details(storage.pg)
        assert isinstance(result, list)

    async def test_each_row_has_required_fields(self, storage):
        for row in await get_per_work_details(storage.pg):
            for key in ("work_id", "title", "author", "source_class",
                        "chunk_count", "embedded_count", "embedding_pct"):
                assert key in row, f"Missing key: {key}"

    async def test_embedding_pct_in_range(self, storage):
        for row in await get_per_work_details(storage.pg):
            assert isinstance(row["embedding_pct"], float)
            assert 0.0 <= row["embedding_pct"] <= 100.0


@SKIP_NO_DB
class TestGetGraphStats:
    async def test_returns_dict(self, storage):
        result = await get_graph_stats(storage.neo4j)
        assert isinstance(result, dict)

    async def test_has_expected_keys_when_neo4j_up(self, storage):
        result = await get_graph_stats(storage.neo4j)
        if result.get("error") is None:
            for key in ("node_counts", "edge_counts", "total_nodes",
                        "total_edges", "top_themes"):
                assert key in result

    async def test_top_themes_is_list(self, storage):
        result = await get_graph_stats(storage.neo4j)
        if result.get("error") is None:
            assert isinstance(result["top_themes"], list)

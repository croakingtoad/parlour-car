"""Unit tests for graph-wide theme coverage in dashboard stats."""

import json

from starlette.applications import Starlette
from starlette.requests import Request

from author_library.dashboard.endpoint import handle_stats
from author_library.dashboard.queries import get_per_work_theme_counts


class FakeNeo4j:
    """Minimal Neo4j fake keyed by the dashboard query shape."""

    async def execute_read(self, query: str, parameters: object = None) -> list[dict[str, object]]:
        del parameters
        if "themed_chunk_count" in query:
            return [
                {
                    "work_id": "test--with-themes",
                    "themed_chunk_count": 94,
                    "distinct_theme_count": 12,
                }
            ]
        if "labels(n)" in query:
            return []
        if "MATCH (n) RETURN count(n)" in query:
            return [{"cnt": 0}]
        if "MATCH ()-[r]->()" in query:
            return []
        if "top_themes" in query or "canonical_name AS name" in query:
            return []
        raise AssertionError(f"Unexpected Neo4j query: {query}")


class FakePostgres:
    async def fetch_one(self, query: str) -> dict[str, object]:
        if "FROM works" in query:
            return {
                "total_works": 2,
                "primary_works": 0,
                "secondary_works": 0,
                "contextual_works": 0,
                "tertiary_works": 0,
                "personal_works": 0,
                "reference_works": 2,
                "total_words": 0,
                "unique_authors": 1,
                "last_ingestion_date": None,
            }
        if "FROM chunks" in query:
            return {"total_chunks": 100}
        if "FROM chunk_embeddings" in query:
            return {"embedded_chunks": 100}
        if "FROM voice_profiles" in query:
            return {"voice_profile_count": 0}
        raise AssertionError(f"Unexpected PostgreSQL query: {query}")

    async def fetch_all(self, query: str) -> list[dict[str, object]]:
        assert "chunk_embeddings" in query
        return [
            {
                "work_id": "test--with-themes",
                "title": "Themed Work",
                "author": "Test Author",
                "source_class": "reference",
                "ingestion_date": "2026-09-03",
                "classification_confidence": 1.0,
                "chunk_count": 100,
                "embedded_count": 100,
            },
            {
                "work_id": "test--without-themes",
                "title": "Unthemed Work",
                "author": "Test Author",
                "source_class": "primary",
                "ingestion_date": "2026-09-03",
                "classification_confidence": 1.0,
                "chunk_count": 6,
                "embedded_count": 6,
            },
        ]


async def test_theme_counts_use_one_graph_wide_aggregate() -> None:
    result = await get_per_work_theme_counts(FakeNeo4j())  # type: ignore[arg-type]

    assert result == {
        "test--with-themes": {"themed_chunk_count": 94, "distinct_theme_count": 12}
    }


async def test_stats_merges_theme_coverage_and_preserves_zero_theme_works() -> None:
    app = Starlette()
    app.state.dashboard_state = {
        "storage": type("Storage", (), {"pg": FakePostgres(), "neo4j": FakeNeo4j()})()
    }
    request = Request({
        "type": "http",
        "method": "GET",
        "path": "/dashboard/stats",
        "headers": [],
        "query_string": b"",
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
        "app": app,
    })

    response = await handle_stats(request)
    works = {work["work_id"]: work for work in json.loads(response.body)["works"]}

    assert works["test--with-themes"]["theme_coverage_pct"] == 94.0
    assert works["test--with-themes"]["distinct_theme_count"] == 12
    assert works["test--without-themes"]["theme_coverage_pct"] == 0.0
    assert works["test--without-themes"]["distinct_theme_count"] == 0

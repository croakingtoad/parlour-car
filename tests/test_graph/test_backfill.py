"""Tests for PG→Neo4j graph backfill.

Tests the backfill logic that detects works present in PostgreSQL but
missing from Neo4j, and reconstructs their graph nodes/edges.

Unit tests use AsyncMock to verify the correct queries are issued and
the correct logic is followed, without requiring live database connections.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from author_library.graph.backfill import (
    MIRRORED_WORK_FIELDS,
    BackfillResult,
    backfill_missing_graph_data,
    backfill_work_graph,
    check_pg_neo4j_consistency,
    get_neo4j_work_ids,
    get_pg_work_ids,
)

# ---------------------------------------------------------------------------
# Helpers — build mock StorageManager with configurable return values
# ---------------------------------------------------------------------------


def _make_mock_storage(
    *,
    pg_works: list[dict[str, Any]] | None = None,
    neo4j_work_ids: list[str] | None = None,
    neo4j_work_records: list[dict[str, Any]] | None = None,
    pg_chunks: list[dict[str, Any]] | None = None,
    neo4j_chunk_ids: list[str] | None = None,
    pg_chunk_counts: list[dict[str, Any]] | None = None,
    neo4j_chunk_counts: list[dict[str, Any]] | None = None,
    pg_chunk_ids_by_work: dict[str, list[str]] | None = None,
    neo4j_chunk_ids_by_work: dict[str, list[str]] | None = None,
) -> MagicMock:
    """Build a mock StorageManager with async return values.

    This creates a realistic mock that mimics the real StorageManager's
    property-based repository access pattern.
    """
    storage = MagicMock()
    pg_work_by_id = {work["work_id"]: work for work in (pg_works or [])}
    if neo4j_work_records is None:
        neo4j_work_records = [
            (
                {
                    "work_id": work_id,
                    **{
                        field: pg_work_by_id[work_id][field]
                        for field in MIRRORED_WORK_FIELDS
                    },
                }
                if work_id in pg_work_by_id
                else {"work_id": work_id}
            )
            for work_id in (neo4j_work_ids or [])
        ]

    # Mock pg.fetch_all to return different results based on query
    async def _pg_fetch_all(query: str, *args: Any) -> list[Any]:
        if "FROM works" in query:
            return [MagicMock(**{"__iter__": lambda s: iter(w.items()), **{k: v for k, v in w.items()}}) for w in (pg_works or [])]
        if "FROM chunks GROUP BY" in query:
            return [MagicMock(**{k: v for k, v in r.items()}) for r in (pg_chunk_counts or [])]
        return []

    # Use _Record-like objects that support dict() conversion
    work_records = []
    for w in (pg_works or []):
        rec = MagicMock()
        rec.__iter__ = lambda s, w=w: iter(w.items())
        rec.__getitem__ = lambda s, k, w=w: w[k]
        rec.keys = lambda w=w: w.keys()
        rec.values = lambda w=w: w.values()
        rec.items = lambda w=w: w.items()
        work_records.append(rec)

    chunk_count_records = []
    for r in (pg_chunk_counts or []):
        rec = MagicMock()
        rec.__getitem__ = lambda s, k, r=r: r[k]
        rec.keys = lambda r=r: r.keys()
        rec.values = lambda r=r: r.values()
        rec.items = lambda r=r: r.items()
        chunk_count_records.append(rec)

    async def _pg_fetch_all_v2(query: str, *args: Any) -> list[Any]:
        if "FROM works" in query:
            return work_records
        if "array_agg(id::text) AS chunk_ids" in query:
            return [
                {"work_id": work_id, "chunk_ids": chunk_ids}
                for work_id, chunk_ids in (pg_chunk_ids_by_work or {}).items()
            ]
        if "FROM chunks GROUP BY" in query:
            return chunk_count_records
        return []

    storage.pg = MagicMock()
    storage.pg.fetch_all = AsyncMock(side_effect=_pg_fetch_all_v2)

    # Mock Neo4j execute_read for different queries
    async def _neo4j_read(query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if "w.title AS title" in query:
            return neo4j_work_records or []
        if "MATCH (w:Work)" in query and "RETURN w.work_id" in query:
            return [{"work_id": wid} for wid in (neo4j_work_ids or [])]
        if "RETURN c.work_id AS work_id, collect(c.chunk_id) AS chunk_ids" in query:
            return [
                {"work_id": work_id, "chunk_ids": chunk_ids}
                for work_id, chunk_ids in (neo4j_chunk_ids_by_work or {}).items()
            ]
        if "MATCH (c:Chunk" in query and "chunk_id" in query:
            return [{"chunk_id": cid} for cid in (neo4j_chunk_ids or [])]
        if "MATCH (c:Chunk)" in query and "COUNT" in query:
            return neo4j_chunk_counts or []
        return []

    storage.neo4j = MagicMock()
    storage.neo4j.execute_read = AsyncMock(side_effect=_neo4j_read)
    storage.neo4j.execute_write = AsyncMock(return_value=[])

    # Mock graph repository
    storage.graph = MagicMock()
    storage.graph.upsert_work_node = AsyncMock()
    storage.graph.upsert_chunk_node = AsyncMock()

    # Mock chunk repository
    chunk_records = []
    for c in (pg_chunks or []):
        rec = MagicMock()
        rec.__getitem__ = lambda s, k, c=c: c[k]
        rec.get = lambda k, default=None, c=c: c.get(k, default)
        rec.keys = lambda c=c: c.keys()
        rec.values = lambda c=c: c.values()
        rec.items = lambda c=c: c.items()
        chunk_records.append(rec)

    storage.chunks = MagicMock()
    storage.chunks.list_by_work = AsyncMock(return_value=chunk_records)

    # Mock works repository
    storage.works = MagicMock()

    return storage


# ---------------------------------------------------------------------------
# BackfillResult model tests
# ---------------------------------------------------------------------------


class TestBackfillResult:
    """Test the BackfillResult data model."""

    def test_default_values(self) -> None:
        result = BackfillResult()
        assert result.works_checked == 0
        assert result.works_missing == 0
        assert result.works_backfilled == 0
        assert result.chunks_created == 0
        assert result.entities_extracted == 0
        assert result.errors == []

    def test_to_dict(self) -> None:
        result = BackfillResult(
            works_checked=4,
            works_missing=2,
            works_backfilled=2,
            chunks_created=1500,
            entities_extracted=300,
            errors=["one error"],
        )
        d = result.to_dict()
        assert d["works_checked"] == 4
        assert d["works_missing"] == 2
        assert d["works_backfilled"] == 2
        assert d["chunks_created"] == 1500
        assert d["entities_extracted"] == 300
        assert d["errors"] == ["one error"]

    def test_to_dict_empty(self) -> None:
        result = BackfillResult()
        d = result.to_dict()
        assert d["works_checked"] == 0
        assert d["errors"] == []


# ---------------------------------------------------------------------------
# get_pg_work_ids tests
# ---------------------------------------------------------------------------


class TestGetPgWorkIds:
    @pytest.mark.asyncio
    async def test_returns_work_metadata(self) -> None:
        """Should query PG works table and return all work records."""
        works = [
            {"work_id": "guite--faith-hope-poetry", "title": "Faith Hope and Poetry",
             "author": "Test Guite", "source_class": "primary", "publication_year": 2010},
            {"work_id": "guite--word-in-the-wilderness", "title": "Word in the Wilderness",
             "author": "Test Guite", "source_class": "primary", "publication_year": 2014},
        ]
        storage = _make_mock_storage(pg_works=works)
        result = await get_pg_work_ids(storage)

        assert len(result) == 2
        assert result[0]["work_id"] == "guite--faith-hope-poetry"
        assert result[1]["work_id"] == "guite--word-in-the-wilderness"

    @pytest.mark.asyncio
    async def test_returns_empty_for_no_works(self) -> None:
        """Should return empty list when no works exist in PG."""
        storage = _make_mock_storage(pg_works=[])
        result = await get_pg_work_ids(storage)
        assert result == []


# ---------------------------------------------------------------------------
# get_neo4j_work_ids tests
# ---------------------------------------------------------------------------


class TestGetNeo4jWorkIds:
    @pytest.mark.asyncio
    async def test_returns_set_of_ids(self) -> None:
        """Should return a set of work_ids from Neo4j."""
        storage = _make_mock_storage(
            neo4j_work_ids=["guite--faith-hope-poetry", "guite--mariner"]
        )
        result = await get_neo4j_work_ids(storage)
        assert result == {"guite--faith-hope-poetry", "guite--mariner"}

    @pytest.mark.asyncio
    async def test_returns_empty_set_when_no_works(self) -> None:
        """Should return empty set when Neo4j has no Work nodes."""
        storage = _make_mock_storage(neo4j_work_ids=[])
        result = await get_neo4j_work_ids(storage)
        assert result == set()


# ---------------------------------------------------------------------------
# backfill_work_graph tests
# ---------------------------------------------------------------------------


class TestBackfillWorkGraph:
    @pytest.mark.asyncio
    async def test_creates_work_node_and_author_edge(self) -> None:
        """Should create Work node and Author->Work relationship."""
        work = {
            "work_id": "guite--faith-hope-poetry",
            "title": "Faith Hope and Poetry",
            "author": "Test Guite",
            "source_class": "primary",
            "publication_year": 2010,
        }
        storage = _make_mock_storage(pg_chunks=[], neo4j_chunk_ids=[])
        chunks_created, errors = await backfill_work_graph(storage, work)

        # Work node upserted
        storage.graph.upsert_work_node.assert_called_once()
        call_args = storage.graph.upsert_work_node.call_args[0][0]
        assert call_args["work_id"] == "guite--faith-hope-poetry"
        assert call_args["title"] == "Faith Hope and Poetry"

        # Author->Work edge created
        storage.neo4j.execute_write.assert_called_once()
        write_call = storage.neo4j.execute_write.call_args
        assert "MERGE (a:Author {author_id: $author_id})" in write_call[0][0]
        assert write_call[0][1]["author_id"] == "guite"
        assert write_call[0][1]["work_id"] == "guite--faith-hope-poetry"

    @pytest.mark.asyncio
    async def test_creates_chunk_nodes(self) -> None:
        """Should create chunk nodes for each PG chunk missing from Neo4j."""
        work = {
            "work_id": "guite--word-in-the-wilderness",
            "title": "Word in the Wilderness",
            "author": "Test Guite",
            "source_class": "primary",
            "publication_year": 2014,
        }
        pg_chunks = [
            {"id": "chunk-001", "text": "In the beginning was the Word...",
             "granularity": "meso", "source_class": "primary",
             "chapter": "Chapter 1", "section": "Introduction", "position": 0},
            {"id": "chunk-002", "text": "And the Word was with God...",
             "granularity": "meso", "source_class": "primary",
             "chapter": "Chapter 1", "section": "Prologue", "position": 1},
        ]
        storage = _make_mock_storage(
            pg_chunks=pg_chunks,
            neo4j_chunk_ids=[],  # None exist in Neo4j
        )

        chunks_created, errors = await backfill_work_graph(storage, work)

        assert chunks_created == 2
        assert errors == []
        assert storage.graph.upsert_chunk_node.call_count == 2

    @pytest.mark.asyncio
    async def test_skips_existing_chunks(self) -> None:
        """Should not re-create chunks that already exist in Neo4j."""
        work = {
            "work_id": "guite--word-in-the-wilderness",
            "title": "Word in the Wilderness",
            "author": "Test Guite",
            "source_class": "primary",
            "publication_year": 2014,
        }
        pg_chunks = [
            {"id": "chunk-001", "text": "Already in graph",
             "granularity": "meso", "source_class": "primary",
             "chapter": "Ch1", "section": "S1", "position": 0},
            {"id": "chunk-002", "text": "Missing from graph",
             "granularity": "meso", "source_class": "primary",
             "chapter": "Ch1", "section": "S2", "position": 1},
        ]
        storage = _make_mock_storage(
            pg_chunks=pg_chunks,
            neo4j_chunk_ids=["chunk-001"],  # chunk-001 already exists
        )

        chunks_created, errors = await backfill_work_graph(storage, work)

        assert chunks_created == 1  # Only chunk-002 created
        assert storage.graph.upsert_chunk_node.call_count == 1

    @pytest.mark.asyncio
    async def test_handles_zero_pg_chunks(self) -> None:
        """Should return 0 chunks when PG has none for the work."""
        work = {
            "work_id": "guite--empty-work",
            "title": "Empty Work",
            "author": "Test Guite",
            "source_class": "primary",
        }
        storage = _make_mock_storage(pg_chunks=[], neo4j_chunk_ids=[])

        chunks_created, errors = await backfill_work_graph(storage, work)

        assert chunks_created == 0
        assert errors == []

    @pytest.mark.asyncio
    async def test_handles_chunk_creation_error(self) -> None:
        """Should continue processing chunks when one fails."""
        work = {
            "work_id": "guite--problematic",
            "title": "Problematic Work",
            "author": "Test Guite",
            "source_class": "primary",
        }
        pg_chunks = [
            {"id": "chunk-ok", "text": "Good chunk",
             "granularity": "meso", "source_class": "primary",
             "chapter": "Ch1", "section": "S1", "position": 0},
            {"id": "chunk-bad", "text": "Bad chunk",
             "granularity": "meso", "source_class": "primary",
             "chapter": "Ch1", "section": "S2", "position": 1},
        ]
        storage = _make_mock_storage(pg_chunks=pg_chunks, neo4j_chunk_ids=[])

        # First call succeeds, second fails
        call_count = 0
        async def _upsert_with_error(chunk_node: dict) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Neo4j write timeout")

        storage.graph.upsert_chunk_node = AsyncMock(side_effect=_upsert_with_error)

        chunks_created, errors = await backfill_work_graph(storage, work)

        assert chunks_created == 1  # Only first succeeded
        assert len(errors) == 1
        assert "chunk-bad" in errors[0]


# ---------------------------------------------------------------------------
# backfill_missing_graph_data tests
# ---------------------------------------------------------------------------


class TestBackfillMissingGraphData:
    @pytest.mark.asyncio
    async def test_no_works_in_pg(self) -> None:
        """Should return empty result when PG has no works."""
        storage = _make_mock_storage(pg_works=[])
        settings = MagicMock()
        embedding = MagicMock()

        result = await backfill_missing_graph_data(
            storage, embedding, settings, run_entity_extraction=False
        )

        assert result.works_checked == 0
        assert result.works_missing == 0
        assert result.works_backfilled == 0

    @pytest.mark.asyncio
    async def test_all_works_already_in_neo4j(self) -> None:
        """Should report 0 missing when all works are in both stores."""
        works = [
            {"work_id": "guite--faith-hope-poetry", "title": "FHP",
             "author": "Guite", "source_class": "primary", "publication_year": 2010},
            {"work_id": "guite--mariner", "title": "Mariner",
             "author": "Guite", "source_class": "primary", "publication_year": 2017},
        ]
        storage = _make_mock_storage(
            pg_works=works,
            neo4j_work_ids=["guite--faith-hope-poetry", "guite--mariner"],
        )
        settings = MagicMock()
        embedding = MagicMock()

        result = await backfill_missing_graph_data(
            storage, embedding, settings, run_entity_extraction=False
        )

        assert result.works_checked == 2
        assert result.works_missing == 0
        assert result.works_backfilled == 0

    @pytest.mark.asyncio
    async def test_detects_and_backfills_missing_works(self) -> None:
        """Should identify missing works and backfill their graph data."""
        works = [
            {"work_id": "guite--faith-hope-poetry", "title": "FHP",
             "author": "Test Guite", "source_class": "primary", "publication_year": 2010},
            {"work_id": "guite--word-in-the-wilderness", "title": "WitW",
             "author": "Test Guite", "source_class": "primary", "publication_year": 2014},
        ]
        pg_chunks = [
            {"id": "chunk-001", "text": "Sample chunk text",
             "granularity": "meso", "source_class": "primary",
             "chapter": "Ch1", "section": "S1", "position": 0},
        ]
        storage = _make_mock_storage(
            pg_works=works,
            neo4j_work_ids=["guite--faith-hope-poetry"],  # Only one present
            pg_chunks=pg_chunks,
            neo4j_chunk_ids=[],
        )
        settings = MagicMock()
        embedding = MagicMock()

        result = await backfill_missing_graph_data(
            storage, embedding, settings, run_entity_extraction=False
        )

        assert result.works_checked == 2
        assert result.works_missing == 1
        assert result.works_backfilled == 1
        assert result.chunks_created == 1

    @pytest.mark.asyncio
    async def test_entity_extraction_disabled(self) -> None:
        """When run_entity_extraction=False, should skip extraction entirely."""
        works = [
            {"work_id": "guite--missing", "title": "Missing",
             "author": "Guite", "source_class": "primary", "publication_year": 2020},
        ]
        pg_chunks = [
            {"id": "c1", "text": "text", "granularity": "meso",
             "source_class": "primary", "chapter": "Ch", "section": "S", "position": 0},
        ]
        storage = _make_mock_storage(
            pg_works=works,
            neo4j_work_ids=[],
            pg_chunks=pg_chunks,
            neo4j_chunk_ids=[],
        )
        settings = MagicMock()
        embedding = MagicMock()

        result = await backfill_missing_graph_data(
            storage, embedding, settings, run_entity_extraction=False
        )

        assert result.works_backfilled == 1
        assert result.entities_extracted == 0

    @pytest.mark.asyncio
    async def test_entity_extraction_enabled(self) -> None:
        """When run_entity_extraction=True, should invoke entity extraction."""
        works = [
            {"work_id": "guite--missing", "title": "Missing",
             "author": "Test Guite", "source_class": "primary",
             "publication_year": 2020},
        ]
        pg_chunks = [
            {"id": "c1", "text": "Some meaningful text about imagination",
             "granularity": "meso", "source_class": "primary",
             "chapter": "Ch", "section": "S", "position": 0},
        ]
        storage = _make_mock_storage(
            pg_works=works,
            neo4j_work_ids=[],
            pg_chunks=pg_chunks,
            neo4j_chunk_ids=[],
        )
        settings = MagicMock()
        settings.llm.entity_extraction_granularities = "macro,meso,micro"
        settings.api_keys = MagicMock()
        embedding = MagicMock()

        with patch(
            "author_library.graph.backfill._run_entity_extraction_for_work",
            new_callable=AsyncMock,
            return_value=42,
        ) as mock_extract:
            result = await backfill_missing_graph_data(
                storage, embedding, settings, run_entity_extraction=True
            )

        assert result.entities_extracted == 42
        mock_extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_continues_on_work_failure(self) -> None:
        """Should continue processing remaining works if one fails."""
        works = [
            {"work_id": "work-fail", "title": "Fail",
             "author": "Author", "source_class": "primary", "publication_year": 2020},
            {"work_id": "work-ok", "title": "OK",
             "author": "Author", "source_class": "primary", "publication_year": 2021},
        ]
        storage = _make_mock_storage(
            pg_works=works,
            neo4j_work_ids=[],
            pg_chunks=[
                {"id": "c1", "text": "text", "granularity": "meso",
                 "source_class": "primary", "chapter": "Ch", "section": "S",
                 "position": 0},
            ],
            neo4j_chunk_ids=[],
        )
        settings = MagicMock()
        embedding = MagicMock()

        # Make first work fail on graph upsert, second succeed
        call_count = 0
        original_upsert = storage.graph.upsert_work_node

        async def _upsert_fails_first(work_data: dict) -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Connection lost")

        storage.graph.upsert_work_node = AsyncMock(side_effect=_upsert_fails_first)

        result = await backfill_missing_graph_data(
            storage, embedding, settings, run_entity_extraction=False
        )

        assert result.works_missing == 2
        assert result.works_backfilled == 1  # Second succeeded
        assert len(result.errors) == 1
        assert "work-fail" in result.errors[0]


# ---------------------------------------------------------------------------
# check_pg_neo4j_consistency tests
# ---------------------------------------------------------------------------


class TestCheckPgNeo4jConsistency:
    @pytest.mark.asyncio
    async def test_fully_consistent(self) -> None:
        """Should report consistent when PG and Neo4j match."""
        works = [
            {"work_id": "w1", "title": "W1", "author": "A",
             "source_class": "primary", "publication_year": 2020},
            {"work_id": "w2", "title": "W2", "author": "A",
             "source_class": "primary", "publication_year": 2021},
        ]
        storage = _make_mock_storage(
            pg_works=works,
            neo4j_work_ids=["w1", "w2"],
            pg_chunk_counts=[
                {"work_id": "w1", "chunk_count": 100},
                {"work_id": "w2", "chunk_count": 200},
            ],
            neo4j_chunk_counts=[
                {"work_id": "w1", "chunk_count": 100},
                {"work_id": "w2", "chunk_count": 200},
            ],
        )

        report = await check_pg_neo4j_consistency(storage)

        assert report["pg_work_count"] == 2
        assert report["neo4j_work_count"] == 2
        assert report["missing_from_neo4j"] == []
        assert report["extra_in_neo4j"] == []
        assert report["work_property_delta"] == []
        assert report["is_consistent"] is True

    @pytest.mark.asyncio
    async def test_missing_works_detected(self) -> None:
        """Should detect works in PG but missing from Neo4j."""
        works = [
            {"work_id": "w1", "title": "W1", "author": "A",
             "source_class": "primary", "publication_year": 2020},
            {"work_id": "w2", "title": "W2", "author": "A",
             "source_class": "primary", "publication_year": 2021},
        ]
        storage = _make_mock_storage(
            pg_works=works,
            neo4j_work_ids=["w1"],  # w2 missing
            pg_chunk_counts=[
                {"work_id": "w1", "chunk_count": 100},
                {"work_id": "w2", "chunk_count": 200},
            ],
            neo4j_chunk_counts=[
                {"work_id": "w1", "chunk_count": 100},
            ],
        )

        report = await check_pg_neo4j_consistency(storage)

        assert report["missing_from_neo4j"] == ["w2"]
        assert report["is_consistent"] is False

    @pytest.mark.asyncio
    async def test_extra_neo4j_works_detected(self) -> None:
        """Should detect works in Neo4j but not in PG (stale graph data)."""
        works = [
            {"work_id": "w1", "title": "W1", "author": "A",
             "source_class": "primary", "publication_year": 2020},
        ]
        storage = _make_mock_storage(
            pg_works=works,
            neo4j_work_ids=["w1", "w-stale"],  # w-stale not in PG
            pg_chunk_counts=[
                {"work_id": "w1", "chunk_count": 100},
            ],
            neo4j_chunk_counts=[
                {"work_id": "w1", "chunk_count": 100},
                {"work_id": "w-stale", "chunk_count": 50},
            ],
        )

        report = await check_pg_neo4j_consistency(storage)

        assert report["extra_in_neo4j"] == ["w-stale"]
        assert report["is_consistent"] is False

    @pytest.mark.asyncio
    async def test_chunk_count_mismatch(self) -> None:
        """Should flag chunk count mismatches for works in both stores."""
        works = [
            {"work_id": "w1", "title": "W1", "author": "A",
             "source_class": "primary", "publication_year": 2020},
        ]
        storage = _make_mock_storage(
            pg_works=works,
            neo4j_work_ids=["w1"],
            pg_chunk_counts=[
                {"work_id": "w1", "chunk_count": 1174},
            ],
            neo4j_chunk_counts=[
                {"work_id": "w1", "chunk_count": 500},
            ],
        )

        report = await check_pg_neo4j_consistency(storage)

        assert report["pg_work_count"] == 1
        assert report["neo4j_work_count"] == 1
        assert report["missing_from_neo4j"] == []
        # Chunk counts don't match
        assert report["is_consistent"] is False
        chunk_info = report["chunk_counts"][0]
        assert chunk_info["pg_chunks"] == 1174
        assert chunk_info["neo4j_chunks"] == 500
        assert chunk_info["in_sync"] is False

    @pytest.mark.asyncio
    async def test_equal_counts_with_different_chunk_ids_are_inconsistent(self) -> None:
        """Should detect equal-and-opposite chunk identity drift."""
        work_id = "author--drifted-work"
        shared_pg_id = "00000000-0000-0000-0000-000000000001"
        shared_neo4j_id = "00000000000000000000000000000001"
        pg_only_ids = [
            "10000000-0000-0000-0000-000000000001",
            "10000000-0000-0000-0000-000000000002",
        ]
        neo4j_only_ids = [
            "20000000000000000000000000000001",
            "20000000000000000000000000000002",
        ]
        storage = _make_mock_storage(
            pg_works=[{
                "work_id": work_id,
                "title": "Drifted Work",
                "author": "Author",
                "source_class": "primary",
                "publication_year": 2026,
            }],
            neo4j_work_ids=[work_id],
            pg_chunk_counts=[{"work_id": work_id, "chunk_count": 3}],
            neo4j_chunk_counts=[{"work_id": work_id, "chunk_count": 3}],
            pg_chunk_ids_by_work={work_id: [shared_pg_id, *pg_only_ids]},
            neo4j_chunk_ids_by_work={work_id: [shared_neo4j_id, *neo4j_only_ids]},
        )

        report = await check_pg_neo4j_consistency(storage)

        assert report["is_consistent"] is False
        chunk_info = report["chunk_counts"][0]
        assert chunk_info["pg_chunks"] == chunk_info["neo4j_chunks"] == 3
        assert chunk_info["in_sync"] is False
        assert chunk_info["pg_only_chunk_count"] == 2
        assert chunk_info["neo4j_only_chunk_count"] == 2
        assert chunk_info["pg_only_chunk_ids_sample"] == pg_only_ids
        assert chunk_info["neo4j_only_chunk_ids_sample"] == neo4j_only_ids

    @pytest.mark.asyncio
    async def test_chunk_identity_samples_are_bounded(self) -> None:
        """Should cap identity drift samples while reporting complete counts."""
        work_id = "author--large-drift"
        pg_only_ids = [f"pg-{index:02d}" for index in range(25)]
        neo4j_only_ids = [f"neo4j-{index:02d}" for index in range(25)]
        storage = _make_mock_storage(
            pg_works=[{
                "work_id": work_id,
                "title": "Large Drift",
                "author": "Author",
                "source_class": "primary",
                "publication_year": 2026,
            }],
            neo4j_work_ids=[work_id],
            pg_chunk_counts=[{"work_id": work_id, "chunk_count": 25}],
            neo4j_chunk_counts=[{"work_id": work_id, "chunk_count": 25}],
            pg_chunk_ids_by_work={work_id: pg_only_ids},
            neo4j_chunk_ids_by_work={work_id: neo4j_only_ids},
        )

        report = await check_pg_neo4j_consistency(storage)

        chunk_info = report["chunk_counts"][0]
        assert chunk_info["pg_only_chunk_count"] == 25
        assert chunk_info["neo4j_only_chunk_count"] == 25
        assert chunk_info["pg_only_chunk_ids_sample"] == pg_only_ids[:20]
        assert chunk_info["neo4j_only_chunk_ids_sample"] == neo4j_only_ids[:20]

    @pytest.mark.asyncio
    async def test_empty_stores(self) -> None:
        """Should handle both stores being empty."""
        storage = _make_mock_storage(
            pg_works=[],
            neo4j_work_ids=[],
            pg_chunk_counts=[],
            neo4j_chunk_counts=[],
        )

        report = await check_pg_neo4j_consistency(storage)

        assert report["pg_work_count"] == 0
        assert report["neo4j_work_count"] == 0
        assert report["work_property_delta"] == []
        assert report["is_consistent"] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field", MIRRORED_WORK_FIELDS)
    async def test_mirrored_work_property_drift_is_reported(self, field: str) -> None:
        """Each PostgreSQL-authoritative Work property is compared exactly."""
        work = {
            "work_id": "author--drifted-work",
            "title": "Correct Title",
            "author": "Correct Author",
            "source_class": "primary",
            "publication_year": 2024,
        }
        neo4j_work = work.copy()
        neo4j_work[field] = 2025 if field == "publication_year" else f"drifted {field}"
        storage = _make_mock_storage(
            pg_works=[work],
            neo4j_work_ids=[work["work_id"]],
            neo4j_work_records=[neo4j_work],
        )

        report = await check_pg_neo4j_consistency(storage)

        assert report["is_consistent"] is False
        assert report["work_property_delta"] == [
            {"work_id": work["work_id"], "mismatched_properties": [field]}
        ]

    @pytest.mark.asyncio
    async def test_missing_neo4j_work_property_is_reported(self) -> None:
        """A missing property is drift even though the Work ID is shared."""
        work = {
            "work_id": "author--missing-property",
            "title": "Correct Title",
            "author": "Correct Author",
            "source_class": "primary",
            "publication_year": 2024,
        }
        neo4j_work = work.copy()
        del neo4j_work["publication_year"]
        storage = _make_mock_storage(
            pg_works=[work],
            neo4j_work_ids=[work["work_id"]],
            neo4j_work_records=[neo4j_work],
        )

        report = await check_pg_neo4j_consistency(storage)

        assert report["work_property_delta"] == [
            {
                "work_id": work["work_id"],
                "mismatched_properties": ["publication_year"],
            }
        ]

    @pytest.mark.asyncio
    async def test_missing_and_extra_works_are_not_property_delta_rows(self) -> None:
        """Property differences apply only to the shared Work-ID intersection."""
        shared_work = {
            "work_id": "author--shared",
            "title": "Shared",
            "author": "Author",
            "source_class": "primary",
            "publication_year": 2024,
        }
        missing_work = {**shared_work, "work_id": "author--missing"}
        storage = _make_mock_storage(
            pg_works=[shared_work, missing_work],
            neo4j_work_ids=[shared_work["work_id"], "author--extra"],
            neo4j_work_records=[
                shared_work,
                {
                    "work_id": "author--extra",
                    "title": "Extra",
                    "author": "Other",
                    "source_class": "primary",
                    "publication_year": 2020,
                },
            ],
        )

        report = await check_pg_neo4j_consistency(storage)

        assert report["missing_from_neo4j"] == ["author--missing"]
        assert report["extra_in_neo4j"] == ["author--extra"]
        assert report["work_property_delta"] == []

    @pytest.mark.asyncio
    async def test_neo4j_work_property_read_failure_is_propagated(self) -> None:
        """The audit handler receives Neo4j Work projection read failures."""
        storage = _make_mock_storage()
        storage.neo4j.execute_read.side_effect = RuntimeError("Neo4j unavailable")

        with pytest.raises(RuntimeError, match="Neo4j unavailable"):
            await check_pg_neo4j_consistency(storage)


# ---------------------------------------------------------------------------
# Entity extraction integration tests
# ---------------------------------------------------------------------------


class TestEntityExtractionBackfill:
    """Test the _run_entity_extraction_for_work helper."""

    @pytest.mark.asyncio
    async def test_skips_when_no_eligible_chunks(self) -> None:
        """Should return 0 when all chunks are nano (not extraction-eligible)."""
        from author_library.graph.backfill import _run_entity_extraction_for_work

        work = {
            "work_id": "guite--test",
            "title": "Test",
            "author": "Test Guite",
            "source_class": "primary",
        }
        pg_chunks = [
            {"id": "c1", "text": "nano text", "granularity": "nano",
             "source_class": "primary", "chapter": "Ch", "section": "S",
             "position": 0, "annotation": None, "work_id": "guite--test"},
        ]
        storage = _make_mock_storage(pg_chunks=pg_chunks)
        settings = MagicMock()
        settings.llm.entity_extraction_granularities = "macro,meso,micro"

        result = await _run_entity_extraction_for_work(storage, work, settings)
        assert result == 0

    @pytest.mark.asyncio
    async def test_skips_when_no_chunks(self) -> None:
        """Should return 0 when work has no chunks at all."""
        from author_library.graph.backfill import _run_entity_extraction_for_work

        work = {
            "work_id": "guite--empty",
            "title": "Empty",
            "author": "Guite",
            "source_class": "primary",
        }
        storage = _make_mock_storage(pg_chunks=[])
        settings = MagicMock()
        settings.llm.entity_extraction_granularities = "macro,meso"

        result = await _run_entity_extraction_for_work(storage, work, settings)
        assert result == 0

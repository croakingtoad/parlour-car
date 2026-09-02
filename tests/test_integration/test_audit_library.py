"""Integration tests for the audit_library MCP tool.

Verifies QG3: on-demand library health check that reports:
- Per-work coverage (chunks, embeddings, entities)
- PG/Neo4j consistency
- Theme graph quality
- Classification anomalies
- Actionable recommendations

Data is inserted directly via storage repositories — no LLM calls needed.
Runs against the test database (author_library_test).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

from author_library.tools.meta import handle_audit_library

from .conftest import SKIP_NO_DB

if TYPE_CHECKING:
    import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_work(
    work_id: str,
    title: str,
    author: str = "Test Author",
    source_class: str = "primary",
    subject_author_id: str = "test-author",
) -> dict[str, Any]:
    return {
        "work_id": work_id,
        "title": title,
        "author": author,
        "source_class": source_class,
        "source_class_note": "Test data for audit_library verification",
        "publication_year": 2000,
        "publisher": "Test Publisher",
        "format_ingested": "txt",
        "word_count": 1000,
        "genre_tags": ["poetry"],
        "subject_headings": [],
        "source_metadata": {
            "subject_author_id": subject_author_id,
            "voice_profile_eligible": True,
        },
    }


def _make_chunk(
    work_id: str,
    text: str,
    position: int = 0,
    granularity: str = "meso",
    source_class: str = "primary",
) -> dict[str, Any]:
    return {
        "work_id": work_id,
        "text": text,
        "annotation": None,
        "granularity": granularity,
        "source_class": source_class,
        "chapter": None,
        "section": None,
        "position": position,
        "parent_chunk_id": None,
        "metadata": {},
    }


async def _run_targeted_audit(
    clean_storage: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    noise_count: int = 0,
    orphan_count: int = 0,
    anomalies: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run an otherwise healthy audit with one targeted warning source."""
    work_id = "test--audit-targeted"

    async def fetch_all(query: str, *_args: Any) -> list[dict[str, Any]]:
        if "FROM works ORDER BY work_id" in query:
            return [
                {
                    "work_id": work_id,
                    "title": "Targeted Audit Work",
                    "source_class": "primary",
                    "author": "Test Author",
                }
            ]
        if "COUNT(*) AS chunk_count" in query:
            return [{"work_id": work_id, "chunk_count": 10}]
        if "COUNT(DISTINCT ce.chunk_id)" in query:
            return [{"work_id": work_id, "embedded_chunks": 10}]
        if "length(text) < 50" in query:
            return [{"work_id": work_id, "noise_count": noise_count}] if noise_count else []
        if "source_metadata->>'subject_author_id'" in query:
            return anomalies or []
        raise AssertionError(f"Unexpected PG query: {query}")

    async def execute_read(query: str, *_args: Any) -> list[dict[str, Any]]:
        if "COUNT(r) AS entity_edges" in query:
            return [{"work_id": work_id, "entity_edges": 1}]
        if "COUNT(c) AS orphan_count" in query:
            return [{"work_id": work_id, "orphan_count": orphan_count}] if orphan_count else []
        if "MATCH (t:Theme)" in query or "WHERE n:Person" in query:
            return []
        raise AssertionError(f"Unexpected Neo4j query: {query}")

    consistency = {
        "is_consistent": True,
        "pg_work_count": 1,
        "neo4j_work_count": 1,
        "missing_from_neo4j": [],
        "extra_in_neo4j": [],
        "chunk_counts": [],
    }
    monkeypatch.setattr(clean_storage.pg, "fetch_all", AsyncMock(side_effect=fetch_all))
    monkeypatch.setattr(clean_storage.neo4j, "execute_read", AsyncMock(side_effect=execute_read))
    monkeypatch.setattr(
        "author_library.graph.backfill.check_pg_neo4j_consistency",
        AsyncMock(return_value=consistency),
    )

    return json.loads(await handle_audit_library({}, storage=clean_storage))


# ---------------------------------------------------------------------------
# TestAuditLibraryEmpty
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestAuditLibraryEmpty:
    """audit_library handles an empty library gracefully."""

    async def test_empty_library_returns_healthy(self, clean_storage: Any) -> None:
        """audit_library on empty DB returns healthy with empty works list."""
        result_str = await handle_audit_library({}, storage=clean_storage)
        result = json.loads(result_str)

        assert "overall_status" in result
        assert "works" in result
        assert "recommendations" in result
        assert result["works"] == []
        # Empty library is healthy (nothing to break)
        assert result["overall_status"] == "healthy"


# ---------------------------------------------------------------------------
# TestAuditLibraryResponseStructure
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestAuditLibraryResponseStructure:
    """audit_library returns the expected top-level structure."""

    async def test_response_has_required_keys(self, clean_storage: Any) -> None:
        """audit_library response always includes all required top-level keys."""
        await clean_storage.works.create(_make_work("test--audit-struct", "Structure Test Work"))
        await clean_storage.chunks.create(
            _make_chunk("test--audit-struct", "A test chunk for structure verification.", 0)
        )

        result_str = await handle_audit_library({}, storage=clean_storage)
        result = json.loads(result_str)

        assert "overall_status" in result, "Missing overall_status key"
        assert "works" in result, "Missing works key"
        assert "graph" in result, "Missing graph key"
        assert "pg_neo4j" in result, "Missing pg_neo4j key"
        assert "recommendations" in result, "Missing recommendations key"

    async def test_overall_status_is_valid_value(self, clean_storage: Any) -> None:
        """overall_status is one of: healthy, warnings, errors."""
        await clean_storage.works.create(_make_work("test--audit-status", "Status Test Work"))

        result_str = await handle_audit_library({}, storage=clean_storage)
        result = json.loads(result_str)

        assert result["overall_status"] in ("healthy", "warnings", "errors"), (
            f"Invalid overall_status: {result['overall_status']!r}"
        )

    async def test_recommendations_is_list(self, clean_storage: Any) -> None:
        """recommendations is always a list (never None)."""
        result_str = await handle_audit_library({}, storage=clean_storage)
        result = json.loads(result_str)
        assert isinstance(result["recommendations"], list)


# ---------------------------------------------------------------------------
# TestAuditLibraryPerWorkStats
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestAuditLibraryPerWorkStats:
    """audit_library returns accurate per-work statistics."""

    async def test_work_entry_has_required_fields(self, clean_storage: Any) -> None:
        """Each work entry includes work_id, title, source_class, chunks, etc."""
        await clean_storage.works.create(_make_work("test--audit-work-fields", "Fields Test"))
        await clean_storage.chunks.create(
            _make_chunk("test--audit-work-fields", "Testing per-work fields.", 0)
        )

        result_str = await handle_audit_library({}, storage=clean_storage)
        result = json.loads(result_str)

        work = next(
            (w for w in result["works"] if w["work_id"] == "test--audit-work-fields"),
            None,
        )
        assert work is not None, "Work entry not found in audit report"
        assert "work_id" in work
        assert "title" in work
        assert "source_class" in work
        assert "chunks" in work
        assert "embeddings" in work
        assert "entities" in work
        assert "warnings" in work

    async def test_chunk_count_reflects_inserted_chunks(self, clean_storage: Any) -> None:
        """audit_library reports the exact chunk count per work."""
        await clean_storage.works.create(_make_work("test--audit-chunk-count", "Chunk Count Test"))
        for i in range(3):
            await clean_storage.chunks.create(
                _make_chunk("test--audit-chunk-count", f"Chunk number {i} for testing.", i)
            )

        result_str = await handle_audit_library({}, storage=clean_storage)
        result = json.loads(result_str)

        work = next(
            (w for w in result["works"] if w["work_id"] == "test--audit-chunk-count"),
            None,
        )
        assert work is not None
        assert work["chunks"] == 3, f"Expected 3 chunks, got {work['chunks']}"

    async def test_work_with_no_chunks_gets_warning(self, clean_storage: Any) -> None:
        """Works with no chunks receive a 'no_chunks' warning."""
        await clean_storage.works.create(_make_work("test--audit-no-chunks", "No Chunks Work"))

        result_str = await handle_audit_library({}, storage=clean_storage)
        result = json.loads(result_str)

        work = next(
            (w for w in result["works"] if w["work_id"] == "test--audit-no-chunks"),
            None,
        )
        assert work is not None
        assert "no_chunks" in work["warnings"], (
            f"Expected 'no_chunks' warning, got: {work['warnings']}"
        )

    async def test_work_with_no_chunks_triggers_error_status(self, clean_storage: Any) -> None:
        """A work with no chunks makes overall_status 'errors'."""
        await clean_storage.works.create(
            _make_work("test--audit-error-status", "Error Status Work")
        )

        result_str = await handle_audit_library({}, storage=clean_storage)
        result = json.loads(result_str)

        assert result["overall_status"] == "errors", (
            f"Expected 'errors' status for work with no chunks, got {result['overall_status']!r}"
        )

    async def test_work_with_no_embeddings_gets_warning(self, clean_storage: Any) -> None:
        """Works with chunks but no embeddings receive a 'no_embeddings' warning."""
        await clean_storage.works.create(_make_work("test--audit-no-embeds", "No Embeddings Work"))
        await clean_storage.chunks.create(
            _make_chunk("test--audit-no-embeds", "A chunk without an embedding.", 0)
        )

        result_str = await handle_audit_library({}, storage=clean_storage)
        result = json.loads(result_str)

        work = next(
            (w for w in result["works"] if w["work_id"] == "test--audit-no-embeds"),
            None,
        )
        assert work is not None
        assert "no_embeddings" in work["warnings"], (
            f"Expected 'no_embeddings' warning, got: {work['warnings']}"
        )

    async def test_multiple_works_all_appear(self, clean_storage: Any) -> None:
        """audit_library reports all ingested works, not just the first."""
        for i in range(3):
            wid = f"test--audit-multi-{i}"
            await clean_storage.works.create(_make_work(wid, f"Multi Work {i}"))
            await clean_storage.chunks.create(
                _make_chunk(wid, f"Text for work {i} in multi-work test.", 0)
            )

        result_str = await handle_audit_library({}, storage=clean_storage)
        result = json.loads(result_str)

        reported_ids = {w["work_id"] for w in result["works"]}
        for i in range(3):
            assert f"test--audit-multi-{i}" in reported_ids, (
                f"test--audit-multi-{i} missing from audit report"
            )


# ---------------------------------------------------------------------------
# TestAuditLibraryPgNeo4j
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestAuditLibraryPgNeo4j:
    """audit_library includes PG/Neo4j consistency section."""

    async def test_pg_neo4j_section_present(self, clean_storage: Any) -> None:
        """pg_neo4j section always present in the response."""
        await clean_storage.works.create(
            _make_work("test--audit-consistency", "Consistency Test Work")
        )

        result_str = await handle_audit_library({}, storage=clean_storage)
        result = json.loads(result_str)

        pg_neo4j = result.get("pg_neo4j", {})
        assert "is_consistent" in pg_neo4j or "error" in pg_neo4j, (
            f"pg_neo4j section missing is_consistent or error: {pg_neo4j}"
        )

    async def test_pg_neo4j_missing_from_neo4j_is_list(self, clean_storage: Any) -> None:
        """pg_neo4j.missing_from_neo4j is a list (may be empty or populated)."""
        await clean_storage.works.create(_make_work("test--audit-missing", "Missing Neo4j Work"))

        result_str = await handle_audit_library({}, storage=clean_storage)
        result = json.loads(result_str)

        pg_neo4j = result.get("pg_neo4j", {})
        if "missing_from_neo4j" in pg_neo4j:
            assert isinstance(pg_neo4j["missing_from_neo4j"], list)


# ---------------------------------------------------------------------------
# TestAuditLibraryRecommendations
# ---------------------------------------------------------------------------


@SKIP_NO_DB
class TestAuditLibraryRecommendations:
    """audit_library provides actionable recommendations."""

    async def test_healthy_library_gets_healthy_recommendation(self, clean_storage: Any) -> None:
        """Empty library reports healthy with a 'no issues' recommendation."""
        result_str = await handle_audit_library({}, storage=clean_storage)
        result = json.loads(result_str)

        # Either no works (trivially healthy) or explicit healthy message
        recs = result["recommendations"]
        assert isinstance(recs, list)
        assert len(recs) > 0, "Expected at least one recommendation"

    async def test_no_embeddings_generates_recommendation(self, clean_storage: Any) -> None:
        """Works without embeddings appear in recommendations."""
        await clean_storage.works.create(
            _make_work("test--audit-rec-embeds", "Recommendation Embeddings Work")
        )
        await clean_storage.chunks.create(
            _make_chunk(
                "test--audit-rec-embeds", "A chunk without embeddings for recommendations.", 0
            )
        )

        result_str = await handle_audit_library({}, storage=clean_storage)
        result = json.loads(result_str)

        recs = result["recommendations"]
        has_embed_rec = any("embed" in r.lower() for r in recs)
        assert has_embed_rec, f"Expected embedding recommendation, got: {recs}"

    async def test_excessive_noise_never_claims_library_is_healthy(
        self, clean_storage: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Noise above the warning threshold produces actionable advice."""
        result = await _run_targeted_audit(clean_storage, monkeypatch, noise_count=2)

        assert result["overall_status"] == "warnings"
        assert not any("library is healthy" in rec.lower() for rec in result["recommendations"])
        assert any(
            "noise chunks" in rec.lower() and "re-chunk" in rec.lower()
            for rec in result["recommendations"]
        )

    async def test_entity_edge_gap_generates_safe_recommendation(
        self, clean_storage: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Material entity-edge gaps produce the non-destructive backfill advice."""
        result = await _run_targeted_audit(clean_storage, monkeypatch, orphan_count=2)

        assert result["overall_status"] == "warnings"
        assert any(
            "entity-edge" in rec.lower()
            and "extraction coverage" in rec.lower()
            and "scripts/backfill_graph_and_entities.py" in rec
            for rec in result["recommendations"]
        )

    async def test_classification_anomaly_recommends_review_and_reclassification(
        self, clean_storage: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Classification anomalies explain the corrective next step."""
        anomaly = {
            "work_id": "test--audit-targeted",
            "title": "Targeted Audit Work",
            "author": "Test Author",
            "source_class": "secondary",
            "subject_author_id": "Test Author",
        }
        result = await _run_targeted_audit(clean_storage, monkeypatch, anomalies=[anomaly])

        assert result["overall_status"] == "warnings"
        assert any(
            "classification anomaly" in rec.lower() and "reclassify" in rec.lower()
            for rec in result["recommendations"]
        )

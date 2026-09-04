"""Tests for meta tool handlers — input validation."""

from __future__ import annotations

import json
import re
import shlex
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from scripts.backfill_graph_and_entities import (
    DESTRUCTIVE_GRAPH_STEPS,
    build_execution_plan,
    parse_args,
)

from author_library.errors import RetrievalError
from author_library.tools.meta import (
    ENTITY_EDGE_GAP_WARNING_MIN_CHUNKS,
    ENTITY_EDGE_GAP_WARNING_MIN_SHARE,
    _classify_chunk_noise,
    _entity_edge_gap_is_warning,
    _format_year_range,
    handle_audit_library,
    handle_author_bio,
    handle_list_works,
)

if TYPE_CHECKING:
    from pytest import MonkeyPatch


async def _run_targeted_audit(
    monkeypatch: MonkeyPatch,
    *,
    chunk_count: int = 100,
    noise_count: int = 0,
    orphan_count: int = 0,
    consistency: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run an otherwise healthy audit with targeted graph findings."""
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
            return [{"work_id": work_id, "chunk_count": chunk_count}]
        if "COUNT(DISTINCT ce.chunk_id)" in query:
            return [{"work_id": work_id, "embedded_chunks": chunk_count}]
        if "length(text) < 50" in query:
            if noise_count:
                return [{"work_id": work_id, "noise_count": noise_count}]
            return []
        if "source_metadata->>'subject_author_id'" in query:
            return []
        raise AssertionError(f"Unexpected PG query: {query}")

    async def execute_read(query: str, *_args: Any) -> list[dict[str, Any]]:
        if "COUNT(r) AS entity_edges" in query:
            return [{"work_id": work_id, "entity_edges": max(1, chunk_count - orphan_count)}]
        if "COUNT(c) AS orphan_count" in query:
            if orphan_count:
                return [{"work_id": work_id, "orphan_count": orphan_count}]
            return []
        if "MATCH (t:Theme)" in query or "WHERE n:Person" in query:
            return []
        raise AssertionError(f"Unexpected Neo4j query: {query}")

    if consistency is None:
        consistency = {
            "is_consistent": True,
            "pg_work_count": 1,
            "neo4j_work_count": 1,
            "missing_from_neo4j": [],
            "extra_in_neo4j": [],
            "chunk_counts": [
                {
                    "work_id": work_id,
                    "pg_chunks": chunk_count,
                    "neo4j_chunks": chunk_count,
                    "in_sync": True,
                    "pg_only_chunk_count": 0,
                    "neo4j_only_chunk_count": 0,
                    "pg_only_chunk_ids_sample": [],
                    "neo4j_only_chunk_ids_sample": [],
                }
            ],
        }

    storage = SimpleNamespace(
        pg=SimpleNamespace(fetch_all=AsyncMock(side_effect=fetch_all)),
        neo4j=SimpleNamespace(execute_read=AsyncMock(side_effect=execute_read)),
    )
    monkeypatch.setattr(
        "author_library.graph.backfill.check_pg_neo4j_consistency",
        AsyncMock(return_value=consistency),
    )

    return json.loads(await handle_audit_library({}, storage=storage))


class TestClassifyChunkNoise:
    """Classify short chunks without requiring database fixtures."""

    @pytest.mark.parametrize(
        ("noise", "total"),
        [
            (0, 100),
            (1, 0),
        ],
    )
    def test_empty_noise_population_has_no_classification(self, noise: int, total: int) -> None:
        assert _classify_chunk_noise(noise, total, ["prose"]) is None

    def test_prose_at_threshold_has_no_classification(self) -> None:
        assert _classify_chunk_noise(10, 100, ["prose"]) is None

    def test_prose_above_threshold_is_warning(self) -> None:
        classification = _classify_chunk_noise(11, 100, ["prose"])

        assert classification is not None
        severity, message = classification
        assert severity == "warning"
        assert "noise_chunks" in message

    @pytest.mark.parametrize(
        ("noise", "total", "genre"),
        [
            (2_392, 2_641, "blessings"),
            (2_215, 14_704, "lecture"),
        ],
    )
    def test_short_line_genre_is_informational(self, noise: int, total: int, genre: str) -> None:
        classification = _classify_chunk_noise(noise, total, [genre])

        assert classification is not None
        severity, message = classification
        assert severity == "info"
        assert "literary form" in message
        assert "excluded from retrieval" in message

    @pytest.mark.parametrize("genre_tags", [[], None])
    def test_missing_genre_uses_default_threshold(self, genre_tags: list[str] | None) -> None:
        classification = _classify_chunk_noise(11, 100, genre_tags)

        assert classification is not None
        assert classification[0] == "warning"

    def test_one_short_line_genre_in_mixed_tags_is_informational(self) -> None:
        classification = _classify_chunk_noise(
            15,
            100,
            ["history", "video-transcript"],
        )

        assert classification is not None
        assert classification[0] == "info"


def test_undated_year_range_has_explicit_label() -> None:
    assert _format_year_range(None, None) == "undated"
    assert _format_year_range(1990, 2020) == "1990-2020"


class TestHandleAuthorBioValidation:
    """Validate required argument checks for author_bio."""

    async def test_missing_author_id_raises(self) -> None:
        with pytest.raises(RetrievalError, match="author_id is required"):
            await handle_author_bio(
                {},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
            )

    async def test_empty_author_id_raises(self) -> None:
        with pytest.raises(RetrievalError, match="author_id is required"):
            await handle_author_bio(
                {"author_id": ""},
                settings=None,  # type: ignore[arg-type]
                storage=None,  # type: ignore[arg-type]
            )


class TestHandleListWorksValidation:
    """Validate required argument checks for list_works."""

    async def test_missing_author_id_raises(self) -> None:
        with pytest.raises(RetrievalError, match="author_id is required"):
            await handle_list_works(
                {},
                storage=None,  # type: ignore[arg-type]
            )


class TestHandleAuditLibraryRecommendations:
    """Translate graph coverage and identity findings into safe actions."""

    async def test_entity_edge_gap_recommends_non_destructive_backfill(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        result = await _run_targeted_audit(
            monkeypatch,
            chunk_count=200,
            orphan_count=ENTITY_EDGE_GAP_WARNING_MIN_CHUNKS,
        )

        recommendations = result["recommendations"]
        assert result["overall_status"] == "warnings"
        assert any(
            "scripts/backfill_graph_and_entities.py" in recommendation
            and "test--audit-targeted" in recommendation
            for recommendation in recommendations
        )
        assert not any(
            "cleanup_neo4j_orphans.py" in recommendation for recommendation in recommendations
        )
        assert not any(
            "library is healthy" in recommendation.lower() for recommendation in recommendations
        )

    async def test_recommended_backfill_execution_plan_cannot_delete_graph(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        result = await _run_targeted_audit(
            monkeypatch,
            chunk_count=200,
            orphan_count=ENTITY_EDGE_GAP_WARNING_MIN_CHUNKS,
        )
        recommendation = next(
            item
            for item in result["recommendations"]
            if "scripts/backfill_graph_and_entities.py" in item
        )
        command_match = re.search(
            r"`(uv run python scripts/backfill_graph_and_entities\.py [^`]+)`",
            recommendation,
        )

        assert command_match is not None
        command = shlex.split(command_match.group(1))
        args = parse_args(command[4:])
        execution_plan = build_execution_plan(args)

        assert command[:4] == [
            "uv",
            "run",
            "python",
            "scripts/backfill_graph_and_entities.py",
        ]
        assert not DESTRUCTIVE_GRAPH_STEPS.intersection(execution_plan)

    async def test_no_entity_edge_gap_has_no_backfill_recommendation(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        result = await _run_targeted_audit(monkeypatch)

        assert result["overall_status"] == "healthy"
        assert not any(
            "backfill_graph_and_entities.py" in recommendation
            for recommendation in result["recommendations"]
        )

    async def test_small_entity_edge_gap_stays_below_warning_threshold(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        result = await _run_targeted_audit(
            monkeypatch,
            chunk_count=100,
            orphan_count=1,
        )

        assert result["overall_status"] == "healthy"
        assert not any(
            "backfill_graph_and_entities.py" in recommendation
            for recommendation in result["recommendations"]
        )

    async def test_high_share_entity_edge_gap_warns_below_absolute_floor(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        result = await _run_targeted_audit(
            monkeypatch,
            chunk_count=10,
            orphan_count=1,
        )

        assert result["overall_status"] == "warnings"
        assert any(
            "scripts/backfill_graph_and_entities.py" in recommendation
            for recommendation in result["recommendations"]
        )

    @pytest.mark.parametrize(
        ("total_chunks", "uncovered_chunks", "expected"),
        [
            pytest.param(
                200,
                ENTITY_EDGE_GAP_WARNING_MIN_CHUNKS,
                True,
                id="exact-absolute-boundary-only",
            ),
            pytest.param(
                10,
                1,
                True,
                id="exact-percentage-boundary-only",
            ),
            pytest.param(100, 9, False, id="just-below-both-boundaries"),
        ],
    )
    def test_entity_edge_gap_threshold_boundaries(
        self,
        total_chunks: int,
        uncovered_chunks: int,
        expected: bool,
    ) -> None:
        assert ENTITY_EDGE_GAP_WARNING_MIN_SHARE == 0.1
        assert _entity_edge_gap_is_warning(total_chunks, uncovered_chunks) is expected

    async def test_identity_drift_recommends_directional_remedies(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        consistency = {
            "is_consistent": False,
            "pg_work_count": 1,
            "neo4j_work_count": 1,
            "missing_from_neo4j": [],
            "extra_in_neo4j": [],
            "chunk_counts": [
                {
                    "work_id": "test--audit-targeted",
                    "pg_chunks": 100,
                    "neo4j_chunks": 100,
                    "in_sync": False,
                    "pg_only_chunk_count": 2,
                    "neo4j_only_chunk_count": 3,
                    "pg_only_chunk_ids_sample": ["pg-only-1", "pg-only-2"],
                    "neo4j_only_chunk_ids_sample": [
                        "neo4j-only-1",
                        "neo4j-only-2",
                        "neo4j-only-3",
                    ],
                }
            ],
        }

        result = await _run_targeted_audit(monkeypatch, consistency=consistency)

        recommendations = result["recommendations"]
        assert result["overall_status"] == "warnings"
        assert any(
            "scripts/backfill_graph_and_entities.py" in recommendation
            and "test--audit-targeted" in recommendation
            for recommendation in recommendations
        )
        assert any(
            "scripts/cleanup_neo4j_orphans.py" in recommendation
            and "test--audit-targeted" in recommendation
            and "correct the scope" in recommendation
            and "dry-run" in recommendation
            for recommendation in recommendations
        )
        assert not any(
            "library is healthy" in recommendation.lower() for recommendation in recommendations
        )

    async def test_noise_warning_has_specific_remedy_and_never_claims_health(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        result = await _run_targeted_audit(monkeypatch, noise_count=11)

        assert result["overall_status"] == "warnings"
        assert not any(
            "library is healthy" in recommendation.lower()
            for recommendation in result["recommendations"]
        )
        assert any(
            "noise chunks" in recommendation.lower() and "re-chunk" in recommendation.lower()
            for recommendation in result["recommendations"]
        )

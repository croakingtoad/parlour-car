"""Tests for cross-work thematic evolution analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

from author_library.intelligence.evolution import (
    EvolutionStep,
    ThematicEvolution,
    ThematicEvolutionAnalyzer,
    _dated_primary_work_years,
)
from author_library.intelligence.thematic_index import (
    ThematicAppearance,
    ThematicEntry,
)

if TYPE_CHECKING:
    from author_library.config import Settings
    from author_library.storage.postgres import PostgresPool

from tests.test_intelligence.conftest import (
    insert_sample_data,
    requires_anthropic_key,
)

# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------


class TestEvolutionModels:
    """Test evolution data model validation."""

    def test_evolution_step_creation(self) -> None:
        """An evolution step should validate."""
        step = EvolutionStep(
            work_id="test--faith-hope-and-poetry",
            publication_year=2010,
            summary="Introduces the sacramental imagination framework",
            key_chunk_ids=["chunk-1", "chunk-2"],
            self_reflection=False,
        )
        assert step.publication_year == 2010
        assert not step.self_reflection

    def test_evolution_step_with_self_reflection(self) -> None:
        """An evolution step with self-reflection should validate."""
        step = EvolutionStep(
            work_id="test--mariner",
            publication_year=2017,
            summary="Revises earlier understanding of Imagination distinction",
            key_chunk_ids=["chunk-6"],
            self_reflection=True,
            self_reflection_note=(
                "Author explicitly references his earlier book and notes "
                "his understanding has become more nuanced"
            ),
        )
        assert step.self_reflection is True
        assert step.self_reflection_note is not None

    def test_thematic_evolution_creation(self) -> None:
        """A complete thematic evolution should validate."""
        evolution = ThematicEvolution(
            theme="Sacramental Imagination",
            narrative=(
                "Guite's understanding of sacramental imagination evolves from "
                "a relatively schematic framework in 2010 to a more fluid, "
                "experiential account by 2017."
            ),
            steps=[
                EvolutionStep(
                    work_id="test--faith-hope-and-poetry",
                    publication_year=2010,
                    summary="Initial framework established",
                ),
                EvolutionStep(
                    work_id="test--mariner",
                    publication_year=2017,
                    summary="Framework revisited and refined",
                    self_reflection=True,
                    self_reflection_note="References earlier work explicitly",
                ),
            ],
            develops_from_edges=[
                {
                    "from_chunk_id": "chunk-1",
                    "to_chunk_id": "chunk-6",
                    "relationship_note": "Later passage revises earlier framework",
                }
            ],
        )
        assert len(evolution.steps) == 2
        assert len(evolution.develops_from_edges) == 1

    def test_empty_evolution(self) -> None:
        """Evolution with no clear development should still validate."""
        evolution = ThematicEvolution(
            theme="Minor Theme",
            narrative="Insufficient data to trace evolution.",
        )
        assert evolution.steps == []
        assert evolution.develops_from_edges == []

    def test_undated_works_are_excluded_from_chronology(self) -> None:
        years = _dated_primary_work_years(
            [
                {
                    "work_id": "author--dated",
                    "source_class": "primary",
                    "publication_year": 2001,
                },
                {
                    "work_id": "author--undated",
                    "source_class": "primary",
                    "publication_year": None,
                },
                {
                    "work_id": "critic--dated",
                    "source_class": "secondary",
                    "publication_year": 1999,
                },
            ]
        )
        assert years == {"author--dated": 2001}

    async def test_undated_appearance_is_not_sent_for_evolution_analysis(self) -> None:
        analyzer = object.__new__(ThematicEvolutionAnalyzer)
        chunk_repo = AsyncMock()
        chunk_repo.list_by_work.return_value = [
            {"id": "chunk-1", "text": "Dated passage", "source_class": "primary"}
        ]
        theme = ThematicEntry(
            theme="Attention",
            author_stance="Attention develops over time.",
            appearances=[
                ThematicAppearance(
                    work_id="author--dated",
                    chapters=["One"],
                    treatment_summary="Dated treatment",
                ),
                ThematicAppearance(
                    work_id="author--undated",
                    chapters=["Two"],
                    treatment_summary="Undated treatment",
                ),
            ],
        )
        llm_result = {
            "narrative": "A dated narrative.",
            "steps": [
                {
                    "work_id": "author--dated",
                    "publication_year": 2001,
                    "summary": "Dated treatment",
                },
                {
                    "work_id": "author--undated",
                    "publication_year": 0,
                    "summary": "Must be excluded",
                },
            ],
        }

        with patch.object(analyzer, "_call_anthropic", return_value=llm_result) as call:
            evolution = await analyzer._analyze_theme_evolution(
                theme=theme,
                work_years={"author--dated": 2001},
                chunk_repo=chunk_repo,
            )

        chunk_repo.list_by_work.assert_awaited_once_with(
            "author--dated", granularity="meso"
        )
        assert "author--undated" not in call.await_args.args[0]
        assert [step.work_id for step in evolution.steps] == ["author--dated"]


# ---------------------------------------------------------------------------
# Integration test (requires API key + databases)
# ---------------------------------------------------------------------------


@requires_anthropic_key
async def test_evolution_analysis_integration(
    pg_pool: PostgresPool,
    app_settings: Settings,
) -> None:
    """End-to-end evolution analysis against real API and Neo4j."""
    await insert_sample_data(pg_pool)

    from author_library.config import DatabaseSettings
    from author_library.storage.neo4j import Neo4jConnection
    from author_library.storage.repositories import (
        Neo4jGraphRepository,
        PgChunkRepository,
        PgWorkRepository,
    )

    db_settings = DatabaseSettings()
    neo4j = Neo4jConnection(db_settings)
    await neo4j.connect()
    await neo4j.init_schema()

    try:
        work_repo = PgWorkRepository(pg_pool)
        chunk_repo = PgChunkRepository(pg_pool)
        graph_repo = Neo4jGraphRepository(neo4j)

        # Create chunk nodes in Neo4j for edge creation
        for chunk in await chunk_repo.list_by_work(
            "test--faith-hope-and-poetry", granularity="meso"
        ):
            await graph_repo.upsert_chunk_node(
                {
                    "chunk_id": str(chunk["id"]),
                    "work_id": chunk["work_id"],
                    "text_preview": chunk["text"][:200],
                    "granularity": chunk["granularity"],
                    "source_class": chunk["source_class"],
                }
            )

        # Create a test theme with appearances in both works
        themes = [
            ThematicEntry(
                theme="Imagination and Perception",
                author_stance=(
                    "Imagination is the living power and prime agent of perception"
                ),
                appearances=[
                    ThematicAppearance(
                        work_id="test--faith-hope-and-poetry",
                        chapters=["Chapter 2"],
                        treatment_summary="Introduces the Coleridgean framework",
                    ),
                    ThematicAppearance(
                        work_id="test--mariner",
                        chapters=["Chapter 4"],
                        treatment_summary="Revisits and refines the framework",
                    ),
                ],
            ),
        ]

        analyzer = ThematicEvolutionAnalyzer(app_settings)
        evolutions = await analyzer.analyze(
            author_id="test--guite",
            themes=themes,
            work_repo=work_repo,
            chunk_repo=chunk_repo,
            graph_repo=graph_repo,
        )

        assert len(evolutions) == 1
        evolution = evolutions[0]
        assert evolution.theme == "Imagination and Perception"
        assert evolution.narrative  # non-empty

    finally:
        await neo4j.close()

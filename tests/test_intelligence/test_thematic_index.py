"""Tests for thematic index generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from author_library.intelligence.thematic_index import (
    KeyPassage,
    ThematicAppearance,
    ThematicEntry,
    ThematicIndexGenerator,
    _batch_chunks,
    _format_chunks_for_prompt,
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


class TestThematicModels:
    """Test thematic index Pydantic model validation."""

    def test_thematic_entry_creation(self) -> None:
        """A thematic entry with appearances should validate."""
        entry = ThematicEntry(
            theme="Sacramental Imagination",
            author_stance=(
                "Poetry functions sacramentally as an outward sign "
                "of inward spiritual grace"
            ),
            appearances=[
                ThematicAppearance(
                    work_id="malcolm-guite--faith-hope-and-poetry",
                    chapters=["Chapter 1", "Chapter 3"],
                    treatment_summary="Central argument developed across multiple chapters",
                ),
            ],
            related_themes=["Incarnational Theology", "Poetry as Revelation"],
            key_passages=[
                KeyPassage(
                    chunk_id="abc123",
                    text_excerpt="the poem functions sacramentally",
                    work_id="malcolm-guite--faith-hope-and-poetry",
                ),
            ],
        )
        assert entry.theme == "Sacramental Imagination"
        assert len(entry.appearances) == 1
        assert len(entry.key_passages) == 1

    def test_minimal_thematic_entry(self) -> None:
        """A theme with just name and stance should validate."""
        entry = ThematicEntry(
            theme="Poetry",
            author_stance="Poetry is revelatory",
        )
        assert entry.appearances == []
        assert entry.related_themes == []

    def test_thematic_appearance_with_chapters(self) -> None:
        """Appearance can list specific chapters."""
        app = ThematicAppearance(
            work_id="test-work",
            chapters=["Ch. 1", "Ch. 5", "Ch. 12"],
            treatment_summary="Theme develops across three chapters",
        )
        assert len(app.chapters) == 3


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestBatching:
    """Test chunk batching utilities."""

    def test_batch_small_list(self) -> None:
        """Small list should produce one batch."""
        chunks = [{"text": f"t{i}"} for i in range(5)]
        batches = _batch_chunks(chunks, batch_size=25)
        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_batch_exact_multiple(self) -> None:
        """List that is exact multiple of batch_size."""
        chunks = [{"text": f"t{i}"} for i in range(50)]
        batches = _batch_chunks(chunks, batch_size=25)
        assert len(batches) == 2
        assert len(batches[0]) == 25
        assert len(batches[1]) == 25

    def test_batch_with_remainder(self) -> None:
        """List with remainder should produce extra partial batch."""
        chunks = [{"text": f"t{i}"} for i in range(30)]
        batches = _batch_chunks(chunks, batch_size=25)
        assert len(batches) == 2
        assert len(batches[0]) == 25
        assert len(batches[1]) == 5


class TestFormatChunks:
    """Test chunk formatting for prompts."""

    def test_format_includes_work_id(self) -> None:
        """Formatted output should include work_id."""
        chunks = [
            {
                "work_id": "malcolm-guite--faith",
                "text": "Sample text here.",
                "id": "chunk-1",
            }
        ]
        result = _format_chunks_for_prompt(chunks)
        assert "malcolm-guite--faith" in result
        assert "Sample text here." in result
        assert "chunk-1" in result

    def test_format_includes_chapter(self) -> None:
        """Formatted output should include chapter when present."""
        chunks = [
            {
                "work_id": "test",
                "text": "Text.",
                "chapter": "Chapter 3",
                "id": "c1",
            }
        ]
        result = _format_chunks_for_prompt(chunks)
        assert "Chapter 3" in result


# ---------------------------------------------------------------------------
# Integration test (requires API key)
# ---------------------------------------------------------------------------


@requires_anthropic_key
async def test_thematic_index_generation_integration(
    pg_pool: PostgresPool,
    app_settings: Settings,
) -> None:
    """End-to-end thematic index generation against real API."""
    await insert_sample_data(pg_pool)

    from author_library.storage.repositories import (
        PgChunkRepository,
        PgThematicRepository,
        PgWorkRepository,
    )

    work_repo = PgWorkRepository(pg_pool)
    chunk_repo = PgChunkRepository(pg_pool)
    thematic_repo = PgThematicRepository(pg_pool)

    generator = ThematicIndexGenerator(app_settings)
    themes = await generator.generate(
        author_id="malcolm-guite",
        author_name="Malcolm Guite",
        work_repo=work_repo,
        chunk_repo=chunk_repo,
        thematic_repo=thematic_repo,
    )

    assert len(themes) > 0
    # At minimum, imagination/sacramental themes should be identified
    theme_names = [t.theme.lower() for t in themes]
    assert any(
        "imagination" in name or "sacrament" in name or "poetry" in name
        for name in theme_names
    ), f"Expected core themes, got: {theme_names}"

    # Verify storage
    stored = await thematic_repo.list_entries("malcolm-guite")
    assert len(stored) == len(themes)

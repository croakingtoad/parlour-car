"""Tests for voice profile extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from author_library.errors import IntelligenceError
from author_library.intelligence.voice_profile import (
    VoiceProfile,
    VoiceProfileExtractor,
    _sample_diverse_chunks,
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


class TestVoiceProfileModel:
    """Test VoiceProfile Pydantic model validation."""

    def test_valid_voice_profile(self) -> None:
        """A fully populated voice profile should validate."""
        profile = VoiceProfile(
            author_id="test--guite",
            register="academic but accessible, conversational scholarly",
            sentence_patterns=[
                "favors complex sentences with embedded clauses",
                "alternates between analytical exposition and poetic quotation",
            ],
            vocabulary_tendencies=[
                "frequently uses 'sacramental', 'incarnational'",
                "draws on theological vocabulary naturally",
            ],
            rhetorical_moves=[
                "builds arguments through close reading of poetry",
                "uses typological parallels between texts",
            ],
            characteristic_phrases=[
                "outward and visible sign",
                "the poet's task",
            ],
            humor_style="dry wit, self-deprecating",
            example_passages=["A representative passage of the author's work."],
            confidence=0.85,
        )
        assert profile.author_id == "test--guite"
        assert profile.confidence == 0.85
        assert len(profile.sentence_patterns) == 2

    def test_confidence_bounds(self) -> None:
        """Confidence must be between 0.0 and 1.0."""
        with pytest.raises(ValueError):
            VoiceProfile(
                author_id="test",
                register="test",
                sentence_patterns=[],
                vocabulary_tendencies=[],
                rhetorical_moves=[],
                characteristic_phrases=[],
                confidence=1.5,
            )

    def test_minimal_voice_profile(self) -> None:
        """A voice profile with minimal fields should validate."""
        profile = VoiceProfile(
            author_id="test-author",
            register="informal",
            sentence_patterns=[],
            vocabulary_tendencies=[],
            rhetorical_moves=[],
            characteristic_phrases=[],
            confidence=0.3,
        )
        assert profile.humor_style is None
        assert profile.example_passages == []


# ---------------------------------------------------------------------------
# Chunk sampling tests
# ---------------------------------------------------------------------------


class TestChunkSampling:
    """Test the diverse chunk sampling strategy."""

    def test_sample_returns_all_when_under_limit(self) -> None:
        """Should return all chunks when count <= max."""
        chunks = [{"work_id": f"w{i}", "text": f"t{i}"} for i in range(5)]
        result = _sample_diverse_chunks(chunks, max_count=10)
        assert len(result) == 5

    def test_sample_limits_to_max(self) -> None:
        """Should limit to max_count."""
        chunks = [{"work_id": f"w{i % 3}", "text": f"t{i}"} for i in range(50)]
        result = _sample_diverse_chunks(chunks, max_count=10)
        assert len(result) <= 10

    def test_sample_distributes_across_works(self) -> None:
        """Should sample from multiple works, not just one."""
        chunks = (
            [{"work_id": "w1", "text": f"t{i}", "chapter": f"ch{i}"} for i in range(20)]
            + [{"work_id": "w2", "text": f"t{i}", "chapter": f"ch{i}"} for i in range(20)]
            + [{"work_id": "w3", "text": f"t{i}", "chapter": f"ch{i}"} for i in range(20)]
        )
        result = _sample_diverse_chunks(chunks, max_count=15)
        work_ids = {c["work_id"] for c in result}
        assert len(work_ids) == 3, "Should sample from all three works"

    def test_sample_prefers_different_chapters(self) -> None:
        """Should prefer chunks from different chapters within a work."""
        chunks = [
            {"work_id": "w1", "text": f"t{i}", "chapter": f"ch{i}"}
            for i in range(10)
        ]
        result = _sample_diverse_chunks(chunks, max_count=5)
        chapters = {c["chapter"] for c in result}
        assert len(chapters) == 5, "Should select 5 different chapters"


# ---------------------------------------------------------------------------
# Extraction integration test (requires API key)
# ---------------------------------------------------------------------------


@requires_anthropic_key
async def test_voice_extraction_integration(
    pg_pool: PostgresPool,
    app_settings: Settings,
) -> None:
    """End-to-end voice profile extraction against real API."""
    await insert_sample_data(pg_pool)

    from author_library.storage.repositories import PgChunkRepository, PgWorkRepository

    work_repo = PgWorkRepository(pg_pool)
    chunk_repo = PgChunkRepository(pg_pool)

    extractor = VoiceProfileExtractor(app_settings)
    profile = await extractor.extract(
        author_id="test--guite",
        author_name="Malcolm Guite",
        work_repo=work_repo,
        chunk_repo=chunk_repo,
    )

    assert profile.author_id == "test--guite"
    assert 0.0 <= profile.confidence <= 1.0
    assert profile.register  # non-empty
    assert len(profile.characteristic_phrases) > 0


# ---------------------------------------------------------------------------
# Voice profile section-type filtering tests
# ---------------------------------------------------------------------------


def _make_chunk_dict(
    section_type: str,
    source_class: str = "primary",
    work_id: str = "test--test",
) -> dict:
    """Create a minimal chunk dict as returned by chunk_repo.list_by_work."""
    return {
        "id": f"{section_type}-chunk",
        "work_id": work_id,
        "text": f"Sample text for {section_type}",
        "granularity": "meso",
        "source_class": source_class,
        "position": 0,
        "metadata": {"section_type": section_type},
    }


class TestVoiceProfileSectionTypeFiltering:
    """Tests that voice profiling excludes preface and structural section types."""

    def _extractor(self) -> VoiceProfileExtractor:
        from unittest.mock import MagicMock

        settings = MagicMock()
        settings.api_keys.anthropic = "sk-ant-test"
        return VoiceProfileExtractor(settings)

    async def test_chapter_chunks_included(self) -> None:
        """Chapter meso chunks from eligible works are included in voice profiling."""
        from unittest.mock import AsyncMock

        extractor = self._extractor()
        work = {
            "work_id": "test--faith-hope",
            "source_class": "primary",
            "source_metadata": {"voice_profile_eligible": True},
        }
        work_repo = AsyncMock()
        work_repo.list_by_author.return_value = [work]

        chunk_repo = AsyncMock()
        chunk_repo.list_by_work.return_value = [
            _make_chunk_dict("chapter"),
            _make_chunk_dict("chapter"),
        ]

        chunks = await extractor._gather_eligible_chunks(
            author_id="test--guite",
            work_repo=work_repo,
            chunk_repo=chunk_repo,
        )

        assert len(chunks) == 2
        assert all(c["metadata"]["section_type"] == "chapter" for c in chunks)

    async def test_preface_chunks_excluded(self) -> None:
        """Preface meso chunks are NOT included in voice profiling."""
        from unittest.mock import AsyncMock

        extractor = self._extractor()
        work = {
            "work_id": "test--faith-hope",
            "source_class": "primary",
            "source_metadata": {"voice_profile_eligible": True},
        }
        work_repo = AsyncMock()
        work_repo.list_by_author.return_value = [work]

        chunk_repo = AsyncMock()
        chunk_repo.list_by_work.return_value = [
            _make_chunk_dict("chapter"),
            _make_chunk_dict("preface"),  # should be excluded
            _make_chunk_dict("back_matter"),
        ]

        chunks = await extractor._gather_eligible_chunks(
            author_id="test--guite",
            work_repo=work_repo,
            chunk_repo=chunk_repo,
        )

        assert len(chunks) == 2
        section_types = {c["metadata"]["section_type"] for c in chunks}
        assert "preface" not in section_types
        assert "chapter" in section_types
        assert "back_matter" in section_types

    async def test_structural_sections_excluded(self) -> None:
        """Bibliography, index, toc, and front_matter chunks are excluded."""
        from unittest.mock import AsyncMock

        extractor = self._extractor()
        work = {
            "work_id": "test--faith-hope",
            "source_class": "primary",
            "source_metadata": {"voice_profile_eligible": True},
        }
        work_repo = AsyncMock()
        work_repo.list_by_author.return_value = [work]

        chunk_repo = AsyncMock()
        chunk_repo.list_by_work.return_value = [
            _make_chunk_dict("chapter"),
            _make_chunk_dict("bibliography"),
            _make_chunk_dict("index"),
            _make_chunk_dict("toc"),
            _make_chunk_dict("front_matter"),
        ]

        chunks = await extractor._gather_eligible_chunks(
            author_id="test--guite",
            work_repo=work_repo,
            chunk_repo=chunk_repo,
        )

        assert len(chunks) == 1
        assert chunks[0]["metadata"]["section_type"] == "chapter"

    async def test_metadata_as_json_string(self) -> None:
        """Handles metadata stored as a JSON string (asyncpg JSONB variance)."""
        import json
        from unittest.mock import AsyncMock

        extractor = self._extractor()
        work = {
            "work_id": "test--faith-hope",
            "source_class": "primary",
            "source_metadata": {"voice_profile_eligible": True},
        }
        work_repo = AsyncMock()
        work_repo.list_by_author.return_value = [work]

        chunk_repo = AsyncMock()
        chunk_repo.list_by_work.return_value = [
            {
                "id": "c1",
                "work_id": "test--faith-hope",
                "text": "text",
                "granularity": "meso",
                "source_class": "primary",
                "position": 0,
                "metadata": json.dumps({"section_type": "chapter"}),
            },
            {
                "id": "c2",
                "work_id": "test--faith-hope",
                "text": "text",
                "granularity": "meso",
                "source_class": "primary",
                "position": 1,
                "metadata": json.dumps({"section_type": "preface"}),
            },
        ]

        chunks = await extractor._gather_eligible_chunks(
            author_id="test--guite",
            work_repo=work_repo,
            chunk_repo=chunk_repo,
        )

        # Only chapter included; preface excluded even when metadata is a JSON string
        assert len(chunks) == 1

    async def test_missing_section_type_defaults_to_chapter(self) -> None:
        """Chunks with no section_type in metadata default to 'chapter' (voice-eligible)."""
        from unittest.mock import AsyncMock

        extractor = self._extractor()
        work = {
            "work_id": "test--faith-hope",
            "source_class": "primary",
            "source_metadata": {"voice_profile_eligible": True},
        }
        work_repo = AsyncMock()
        work_repo.list_by_author.return_value = [work]

        chunk_repo = AsyncMock()
        chunk_repo.list_by_work.return_value = [
            {
                "id": "c1",
                "work_id": "test--faith-hope",
                "text": "text",
                "granularity": "meso",
                "source_class": "primary",
                "position": 0,
                "metadata": {},  # no section_type key
            }
        ]

        chunks = await extractor._gather_eligible_chunks(
            author_id="test--guite",
            work_repo=work_repo,
            chunk_repo=chunk_repo,
        )

        assert len(chunks) == 1

    async def test_reference_work_never_enters_eligible_work_ids(
        self,
        pg_pool: PostgresPool,
        app_settings: Settings,
    ) -> None:
        """The work-level primary gate excludes reference works before chunk lookup."""
        from author_library.storage.repositories import PgChunkRepository, PgWorkRepository

        work_repo = PgWorkRepository(pg_pool)
        chunk_repo = PgChunkRepository(pg_pool)
        common_work = {
            "author": "test--voice-author",
            "source_class_note": "Voice eligibility regression fixture",
            "publication_year": 2026,
            "publisher": "Test Publisher",
            "format_ingested": "txt",
            "word_count": 100,
            "genre_tags": ["craft"],
            "subject_headings": ["Prosody"],
        }
        await work_repo.create({
            **common_work,
            "work_id": "test--voice-primary",
            "title": "Primary Work",
            "source_class": "primary",
            "source_metadata": {
                "subject_author_id": "test--voice-author",
                "voice_profile_eligible": True,
            },
        })
        await work_repo.create({
            **common_work,
            "work_id": "test--voice-reference",
            "title": "Reference Work",
            "source_class": "reference",
            "source_metadata": {
                "external_author": "Test Voice Author",
                "reference_type": "craft-handbook",
                "subject_domain": "prosody",
            },
        })
        for work_id in ("test--voice-primary", "test--voice-reference"):
            await chunk_repo.create({
                "work_id": work_id,
                "text": f"Meso content for {work_id}",
                "annotation": None,
                "granularity": "meso",
                # Deliberately primary-labelled: the work-level gate must still
                # make the reference work structurally ineligible.
                "source_class": "primary",
                "position": 0,
                "metadata": {"section_type": "chapter"},
            })

        chunks = await VoiceProfileExtractor(app_settings)._gather_eligible_chunks(
            author_id="test--voice-author",
            work_repo=work_repo,
            chunk_repo=chunk_repo,
        )

        assert {chunk["work_id"] for chunk in chunks} == {"test--voice-primary"}


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestVoiceExtractionErrors:
    """Test error handling in voice profile extraction."""

    async def test_no_api_key_raises_error(self) -> None:
        """Should raise IntelligenceError when API key is missing."""
        from pydantic import SecretStr

        from author_library.config import APIKeySettings, Settings

        # Settings loads keys from .env (env_file) — explicit kwargs are the
        # only reliable way to simulate a missing key regardless of the
        # invoking shell's environment
        settings = Settings()
        settings.api_keys = APIKeySettings(anthropic_api_key=SecretStr(""))
        extractor = VoiceProfileExtractor(settings)

        with pytest.raises(IntelligenceError, match="API key is required"):
            await extractor.extract(
                author_id="test",
                author_name="Test",
                work_repo=_StubWorkRepo([]),  # type: ignore[arg-type]
                chunk_repo=_StubChunkRepo([]),  # type: ignore[arg-type]
            )

    async def test_insufficient_corpus_raises_error(
        self,
        pg_pool: PostgresPool,
    ) -> None:
        """Should raise IntelligenceError when corpus is too small."""
        await insert_sample_data(pg_pool)

        # Remove most chunks to create insufficient corpus
        await pg_pool.execute(
            "DELETE FROM chunks WHERE position > 1"
        )

        from author_library.config import APIKeySettings, Settings

        settings = Settings(
            api_keys=APIKeySettings(anthropic_api_key="sk-fake-key-for-test"),
        )

        from author_library.storage.repositories import PgChunkRepository, PgWorkRepository

        extractor = VoiceProfileExtractor(settings)

        with pytest.raises(IntelligenceError, match="Insufficient primary corpus"):
            await extractor.extract(
                author_id="test--guite",
                author_name="Malcolm Guite",
                work_repo=PgWorkRepository(pg_pool),
                chunk_repo=PgChunkRepository(pg_pool),
            )


# ---------------------------------------------------------------------------
# Stubs for testing error paths (not mock data — just interface stubs)
# ---------------------------------------------------------------------------


class _StubWorkRepo:
    """Minimal work repository for testing error paths."""

    def __init__(self, works: list[dict[str, Any]]) -> None:
        self._works = works

    async def list_by_author(self, author: str) -> list[dict[str, Any]]:
        return self._works

    async def get(self, work_id: str) -> dict[str, Any] | None:
        return None

    async def create(self, work: dict[str, Any]) -> str:
        return work["work_id"]

    async def update(self, work_id: str, fields: dict[str, Any]) -> bool:
        return False

    async def delete(self, work_id: str) -> bool:
        return False


class _StubChunkRepo:
    """Minimal chunk repository for testing error paths."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._chunks = chunks

    async def list_by_work(
        self, work_id: str, *, granularity: str | None = None
    ) -> list[dict[str, Any]]:
        return self._chunks

    async def get(self, chunk_id: Any) -> dict[str, Any] | None:
        return None

    async def create(self, chunk: dict[str, Any]) -> Any:
        return None

    async def delete(self, chunk_id: Any) -> bool:
        return False

    async def delete_by_work(self, work_id: str) -> int:
        return 0

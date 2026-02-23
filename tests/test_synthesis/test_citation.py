"""Tests for O3: Source citation enrichment."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from author_library.synthesis.citation import (
    CitationEnricher,
    CitationReport,
    EnrichedCitation,
)
from author_library.synthesis.gatherer import PersonalReflection
from author_library.synthesis.prompt_engine import (
    SourceCitation,
    SynthesisConfidence,
    SynthesisResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_storage():
    storage = MagicMock()
    storage.pg = MagicMock()
    storage.works = MagicMock()
    storage.neo4j = MagicMock()
    return storage


@pytest.fixture()
def enricher(mock_storage):
    return CitationEnricher(storage=mock_storage)


def _make_reflection(
    chunk_id: str = "ref-1",
    work_id: str = "personal--notes",
    themes: list[str] | None = None,
) -> PersonalReflection:
    return PersonalReflection(
        chunk_id=chunk_id,
        work_id=work_id,
        text="My reflection",
        date_created="2026-01-15T10:30:00",
        granularity="micro",
        metadata={
            "section_type": "my_thoughts",
            "themes": themes or [],
            "source_note": "guite-faith-hope-ch3",
        },
    )


def _make_synthesis(citations: list[SourceCitation]) -> SynthesisResult:
    return SynthesisResult(
        synthesis="You appear to think...",
        sources_used=citations,
        confidence=SynthesisConfidence.DEVELOPING,
        open_tensions=[],
        theme="imagination",
        prompt="What do I think?",
        reflection_count=len(citations),
    )


# ---------------------------------------------------------------------------
# EnrichedCitation tests
# ---------------------------------------------------------------------------


class TestEnrichedCitation:
    def test_creation(self):
        cite = EnrichedCitation(
            capture_id="ref-1",
            note_path="personal--notes",
            excerpt="I think...",
            date="2026-01-15",
            work_id="personal--notes",
            work_title="My Notes",
            author="",
            section_type="my_thoughts",
            themes=["imagination"],
            verified=True,
        )
        assert cite.verified
        assert cite.section_type == "my_thoughts"


# ---------------------------------------------------------------------------
# CitationReport tests
# ---------------------------------------------------------------------------


class TestCitationReport:
    def test_to_dict(self):
        report = CitationReport(
            citations=[
                EnrichedCitation(
                    capture_id="ref-1", note_path="p--notes",
                    excerpt="E1", date="2026-01-15",
                    work_id="w-1", work_title="Title",
                    author="Author", section_type="my_thoughts",
                    themes=["imagination"], verified=True,
                ),
            ],
            total_citations=1,
            verified_count=1,
            unverified_count=0,
            unique_works=1,
            date_span=("2026-01-15", "2026-01-15"),
        )
        d = report.to_dict()
        assert d["total_citations"] == 1
        assert d["verified_count"] == 1
        assert len(d["citations"]) == 1

    def test_no_date_span(self):
        report = CitationReport(
            citations=[], total_citations=0,
            verified_count=0, unverified_count=0,
            unique_works=0,
        )
        assert report.to_dict()["date_span"] is None


# ---------------------------------------------------------------------------
# CitationEnricher tests
# ---------------------------------------------------------------------------


class TestCitationEnricher:
    @pytest.mark.asyncio()
    async def test_enrich_verified(self, enricher, mock_storage):
        """Citation verified when chunk exists in database."""
        mock_storage.pg.fetch_one = AsyncMock(return_value={
            "id": "ref-1", "work_id": "personal--notes",
            "source_class": "personal", "granularity": "micro",
            "chapter": "Ch 3", "metadata": {},
        })
        mock_storage.works.get = AsyncMock(return_value={
            "title": "My Notes", "author": "",
        })
        mock_storage.neo4j.execute_read = AsyncMock(return_value=[])

        reflection = _make_reflection(themes=["imagination"])
        citation = SourceCitation(
            capture_id="ref-1", note_path="personal--notes",
            excerpt="I think...", date="2026-01-15",
        )
        synthesis = _make_synthesis([citation])

        report = await enricher.enrich(synthesis, [reflection])

        assert report.total_citations == 1
        assert report.verified_count == 1
        assert report.citations[0].verified
        assert report.citations[0].chapter == "Ch 3"

    @pytest.mark.asyncio()
    async def test_enrich_unverified(self, enricher, mock_storage):
        """Citation unverified when chunk not in database."""
        mock_storage.pg.fetch_one = AsyncMock(return_value=None)
        mock_storage.works.get = AsyncMock(return_value=None)
        mock_storage.neo4j.execute_read = AsyncMock(return_value=[])

        reflection = _make_reflection()
        citation = SourceCitation(
            capture_id="missing-ref", note_path="",
            excerpt="Some text", date="2026-01-15",
        )
        synthesis = _make_synthesis([citation])

        report = await enricher.enrich(synthesis, [reflection])

        assert report.unverified_count == 1
        assert not report.citations[0].verified

    @pytest.mark.asyncio()
    async def test_themes_from_reflection(self, enricher, mock_storage):
        """Themes come from the reflection metadata."""
        mock_storage.pg.fetch_one = AsyncMock(return_value={
            "id": "ref-1", "work_id": "w-1",
            "source_class": "personal", "granularity": "micro",
            "chapter": None, "metadata": {},
        })
        mock_storage.works.get = AsyncMock(return_value=None)

        reflection = _make_reflection(themes=["imagination", "prayer"])
        citation = SourceCitation(
            capture_id="ref-1", note_path="",
            excerpt="Text", date="2026-01-15",
        )
        synthesis = _make_synthesis([citation])

        report = await enricher.enrich(synthesis, [reflection])

        assert report.citations[0].themes == ["imagination", "prayer"]

    @pytest.mark.asyncio()
    async def test_themes_from_graph_fallback(self, enricher, mock_storage):
        """If reflection has no themes, falls back to graph."""
        mock_storage.pg.fetch_one = AsyncMock(return_value={
            "id": "ref-1", "work_id": "w-1",
            "source_class": "personal", "granularity": "micro",
            "chapter": None, "metadata": {},
        })
        mock_storage.works.get = AsyncMock(return_value=None)
        mock_storage.neo4j.execute_read = AsyncMock(return_value=[
            {"theme": "imagination"},
            {"theme": "prayer"},
        ])

        reflection = _make_reflection(themes=[])  # No themes
        citation = SourceCitation(
            capture_id="ref-1", note_path="",
            excerpt="Text", date="2026-01-15",
        )
        synthesis = _make_synthesis([citation])

        report = await enricher.enrich(synthesis, [reflection])

        assert report.citations[0].themes == ["imagination", "prayer"]

    @pytest.mark.asyncio()
    async def test_multiple_citations(self, enricher, mock_storage):
        """Multiple citations produce correct counts."""
        mock_storage.pg.fetch_one = AsyncMock(side_effect=[
            {"id": "ref-1", "work_id": "w-1",
             "source_class": "personal", "granularity": "micro",
             "chapter": None, "metadata": {}},
            {"id": "ref-2", "work_id": "w-2",
             "source_class": "personal", "granularity": "micro",
             "chapter": None, "metadata": {}},
        ])
        mock_storage.works.get = AsyncMock(side_effect=[
            {"title": "Work A", "author": ""},
            {"title": "Work B", "author": ""},
        ])
        mock_storage.neo4j.execute_read = AsyncMock(return_value=[])

        reflections = [
            _make_reflection(chunk_id="ref-1"),
            _make_reflection(chunk_id="ref-2", work_id="personal--other"),
        ]
        citations = [
            SourceCitation(capture_id="ref-1", note_path="", excerpt="E1", date="2026-01-15"),
            SourceCitation(capture_id="ref-2", note_path="", excerpt="E2", date="2026-02-01"),
        ]
        synthesis = _make_synthesis(citations)

        report = await enricher.enrich(synthesis, reflections)

        assert report.total_citations == 2
        assert report.unique_works == 2
        assert report.date_span == ("2026-01-15", "2026-02-01")

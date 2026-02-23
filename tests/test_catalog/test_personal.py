"""Tests for Personal source class (A1) and media format extensions (A5).

Covers:
- SourceClass.PERSONAL enum validity
- PersonalCatalogEntry model validation
- Personal processing route (PERSONAL_ENRICHMENT)
- MediaType enum and new FormatIngested values
- New CatalogEntry media fields (url, duration, speakers, etc.)
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from author_library.catalog.models import (
    CatalogEntry,
    ClassificationResult,
    FormatIngested,
    MediaType,
    PersonalCatalogEntry,
    ProcessingRoute,
    SourceClass,
    route_for_source_class,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _core_fields(**overrides: object) -> dict:
    """Return a valid set of core CatalogEntry fields with optional overrides."""
    base = {
        "work_id": "marty--reflection-on-guite",
        "title": "Reflection on Guite",
        "author": "Marty",
        "source_class": SourceClass.PERSONAL,
        "source_class_note": "Personal reflection on Faith Hope and Poetry.",
        "publication_year": 2026,
        "publisher": "Self",
        "format_ingested": FormatIngested.TXT,
        "language": "en",
        "word_count": 500,
        "genre_tags": ["personal", "reflection"],
        "subject_headings": ["Imagination--Personal response"],
        "ingestion_date": date(2026, 2, 23),
    }
    base.update(overrides)
    return base


def _personal_fields(**overrides: object) -> dict:
    """Return valid PersonalCatalogEntry fields."""
    fields = _core_fields(
        source_class=SourceClass.PERSONAL,
        user_id="marty",
    )
    fields.update(overrides)
    return fields


# ---------------------------------------------------------------------------
# SourceClass.PERSONAL enum
# ---------------------------------------------------------------------------


class TestSourceClassPersonal:
    def test_personal_is_valid_enum_value(self) -> None:
        assert SourceClass.PERSONAL == "personal"

    def test_personal_in_all_source_classes(self) -> None:
        all_classes = list(SourceClass)
        assert SourceClass.PERSONAL in all_classes

    def test_five_source_classes_total(self) -> None:
        assert len(SourceClass) == 5


# ---------------------------------------------------------------------------
# PersonalCatalogEntry model
# ---------------------------------------------------------------------------


class TestPersonalCatalogEntry:
    def test_valid_personal_entry(self) -> None:
        entry = PersonalCatalogEntry(**_personal_fields())
        assert entry.source_class == SourceClass.PERSONAL
        assert entry.user_id == "marty"

    def test_default_user_id(self) -> None:
        fields = _personal_fields()
        del fields["user_id"]
        entry = PersonalCatalogEntry(**fields)
        assert entry.user_id == "marty"

    def test_custom_user_id(self) -> None:
        entry = PersonalCatalogEntry(**_personal_fields(user_id="other-user"))
        assert entry.user_id == "other-user"

    def test_wrong_source_class_rejected(self) -> None:
        with pytest.raises(ValidationError, match="source_class='personal'"):
            PersonalCatalogEntry(**_personal_fields(source_class=SourceClass.PRIMARY))

    def test_inherits_core_fields(self) -> None:
        entry = PersonalCatalogEntry(**_personal_fields())
        assert entry.work_id == "marty--reflection-on-guite"
        assert entry.title == "Reflection on Guite"
        assert entry.word_count == 500


# ---------------------------------------------------------------------------
# Personal processing route
# ---------------------------------------------------------------------------


class TestPersonalProcessingRoute:
    def test_personal_enrichment_route(self) -> None:
        route = route_for_source_class(SourceClass.PERSONAL)
        assert route == ProcessingRoute.PERSONAL_ENRICHMENT

    def test_personal_enrichment_is_distinct(self) -> None:
        """PERSONAL_ENRICHMENT is a separate route from FULL_ENRICHMENT."""
        assert ProcessingRoute.PERSONAL_ENRICHMENT != ProcessingRoute.FULL_ENRICHMENT

    def test_all_source_classes_have_routes(self) -> None:
        for sc in SourceClass:
            route = route_for_source_class(sc)
            assert isinstance(route, ProcessingRoute)


# ---------------------------------------------------------------------------
# ClassificationResult with PERSONAL
# ---------------------------------------------------------------------------


class TestClassificationResultPersonal:
    def test_personal_high_confidence_stays(self) -> None:
        result = ClassificationResult(
            source_class=SourceClass.PERSONAL,
            confidence=0.95,
            reasoning="User-authored reflection text.",
        )
        assert result.source_class == SourceClass.PERSONAL

    def test_personal_low_confidence_downgrades_to_secondary(self) -> None:
        result = ClassificationResult(
            source_class=SourceClass.PERSONAL,
            confidence=0.55,
            reasoning="Uncertain personal classification.",
        )
        assert result.source_class == SourceClass.SECONDARY
        assert "AUTO-DOWNGRADED TO SECONDARY" in result.reasoning


# ---------------------------------------------------------------------------
# FormatIngested media extensions (A5)
# ---------------------------------------------------------------------------


class TestFormatIngestedMedia:
    def test_video_format(self) -> None:
        assert FormatIngested.VIDEO == "video"

    def test_audio_format(self) -> None:
        assert FormatIngested.AUDIO == "audio"

    def test_transcript_format(self) -> None:
        assert FormatIngested.TRANSCRIPT == "transcript"

    def test_youtube_captions_format(self) -> None:
        assert FormatIngested.YOUTUBE_CAPTIONS == "youtube_captions"

    def test_all_original_formats_still_exist(self) -> None:
        assert FormatIngested.EPUB == "epub"
        assert FormatIngested.PDF == "pdf"
        assert FormatIngested.TXT == "txt"
        assert FormatIngested.HTML == "html"
        assert FormatIngested.DOCX == "docx"

    def test_total_format_count(self) -> None:
        assert len(FormatIngested) == 9


# ---------------------------------------------------------------------------
# MediaType enum (A5)
# ---------------------------------------------------------------------------


class TestMediaType:
    def test_book(self) -> None:
        assert MediaType.BOOK == "book"

    def test_video(self) -> None:
        assert MediaType.VIDEO == "video"

    def test_audio(self) -> None:
        assert MediaType.AUDIO == "audio"

    def test_podcast(self) -> None:
        assert MediaType.PODCAST == "podcast"

    def test_article(self) -> None:
        assert MediaType.ARTICLE == "article"

    def test_five_media_types(self) -> None:
        assert len(MediaType) == 5


# ---------------------------------------------------------------------------
# CatalogEntry media fields (A5)
# ---------------------------------------------------------------------------


class TestCatalogEntryMediaFields:
    def test_url_field(self) -> None:
        entry = CatalogEntry(**_core_fields(url="https://youtube.com/watch?v=abc"))
        assert entry.url == "https://youtube.com/watch?v=abc"

    def test_duration_field(self) -> None:
        entry = CatalogEntry(**_core_fields(duration=3600))
        assert entry.duration == 3600

    def test_speakers_field(self) -> None:
        entry = CatalogEntry(**_core_fields(speakers=["Malcolm Guite", "Rowan Williams"]))
        assert entry.speakers == ["Malcolm Guite", "Rowan Williams"]

    def test_date_published_field(self) -> None:
        entry = CatalogEntry(**_core_fields(date_published=date(2024, 6, 15)))
        assert entry.date_published == date(2024, 6, 15)

    def test_date_consumed_field(self) -> None:
        entry = CatalogEntry(**_core_fields(date_consumed=date(2026, 2, 20)))
        assert entry.date_consumed == date(2026, 2, 20)

    def test_transcript_cached_default_false(self) -> None:
        entry = CatalogEntry(**_core_fields())
        assert entry.transcript_cached is False

    def test_transcript_cached_set_true(self) -> None:
        entry = CatalogEntry(**_core_fields(transcript_cached=True))
        assert entry.transcript_cached is True

    def test_media_type_field(self) -> None:
        entry = CatalogEntry(**_core_fields(media=MediaType.VIDEO))
        assert entry.media == MediaType.VIDEO

    def test_media_defaults_to_none(self) -> None:
        entry = CatalogEntry(**_core_fields())
        assert entry.media is None

    def test_speakers_defaults_to_empty(self) -> None:
        entry = CatalogEntry(**_core_fields())
        assert entry.speakers == []

    def test_all_media_fields_together(self) -> None:
        entry = CatalogEntry(
            **_core_fields(
                url="https://youtube.com/watch?v=example",
                duration=5400,
                speakers=["Malcolm Guite"],
                date_published=date(2023, 1, 1),
                date_consumed=date(2026, 2, 1),
                transcript_cached=True,
                media=MediaType.VIDEO,
                format_ingested=FormatIngested.YOUTUBE_CAPTIONS,
            )
        )
        assert entry.url == "https://youtube.com/watch?v=example"
        assert entry.duration == 5400
        assert entry.speakers == ["Malcolm Guite"]
        assert entry.date_published == date(2023, 1, 1)
        assert entry.date_consumed == date(2026, 2, 1)
        assert entry.transcript_cached is True
        assert entry.media == MediaType.VIDEO
        assert entry.format_ingested == FormatIngested.YOUTUBE_CAPTIONS

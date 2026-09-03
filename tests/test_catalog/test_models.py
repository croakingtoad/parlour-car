"""Tests for catalog metadata schema models.

Covers all validation rules from catalog-schema.md §6, source-class-specific
field requirements, and the classification result default-to-secondary rule.
"""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from author_library.catalog.models import (
    CatalogEntry,
    ClassificationResult,
    ContextualCatalogEntry,
    EngagementFrequency,
    EngagementType,
    FormatIngested,
    OcrQuality,
    PrimaryCatalogEntry,
    ProcessingRoute,
    ReferenceCatalogEntry,
    ReferenceType,
    SecondaryCatalogEntry,
    SecondaryRelationship,
    SourceClass,
    TertiaryCatalogEntry,
    WorkType,
    route_for_source_class,
)

# ---------------------------------------------------------------------------
# Fixtures: valid core fields
# ---------------------------------------------------------------------------


def _core_fields(**overrides: object) -> dict:
    """Return a valid set of core CatalogEntry fields with optional overrides."""
    base = {
        "work_id": "malcolm-guite--faith-hope-and-poetry",
        "title": "Faith, Hope and Poetry",
        "author": "Malcolm Guite",
        "source_class": SourceClass.PRIMARY,
        "source_class_note": "Listed as sole author on title page and personal bibliography.",
        "publication_year": 2012,
        "publisher": "Ashgate",
        "format_ingested": FormatIngested.EPUB,
        "language": "en",
        "word_count": 85000,
        "genre_tags": ["monograph", "literary-criticism"],
        "subject_headings": ["Imagination--Religious aspects--Christianity"],
        "ingestion_date": date(2026, 2, 16),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# CatalogEntry (core) tests
# ---------------------------------------------------------------------------


class TestCatalogEntry:
    def test_valid_entry(self) -> None:
        entry = CatalogEntry(**_core_fields())
        assert entry.work_id == "malcolm-guite--faith-hope-and-poetry"
        assert entry.source_class == SourceClass.PRIMARY

    def test_optional_fields(self) -> None:
        entry = CatalogEntry(
            **_core_fields(
                original_publication_year=2010,
                edition="2nd revised",
                isbn="9781409449874",
                ocr_quality=OcrQuality.NOT_APPLICABLE,
                notes="Important monograph",
            )
        )
        assert entry.original_publication_year == 2010
        assert entry.ocr_quality == OcrQuality.NOT_APPLICABLE

    def test_genre_tags_lowercased(self) -> None:
        entry = CatalogEntry(**_core_fields(genre_tags=["Monograph", "THEOLOGY"]))
        assert entry.genre_tags == ["monograph", "theology"]

    def test_language_lowercased(self) -> None:
        entry = CatalogEntry(**_core_fields(language="EN"))
        assert entry.language == "en"


class TestWorkIdValidation:
    def test_valid_work_id(self) -> None:
        entry = CatalogEntry(**_core_fields(work_id="malcolm-guite--faith-hope-and-poetry"))
        assert entry.work_id == "malcolm-guite--faith-hope-and-poetry"

    def test_work_id_too_long(self) -> None:
        long_id = "a" * 60 + "--" + "b" * 70  # 132 chars > 128
        with pytest.raises(ValidationError, match="at most 128 characters"):
            CatalogEntry(**_core_fields(work_id=long_id))

    def test_work_id_missing_double_hyphen(self) -> None:
        with pytest.raises(ValidationError, match="double-hyphen"):
            CatalogEntry(**_core_fields(work_id="malcolm-guite-faith"))

    def test_work_id_uppercase_rejected(self) -> None:
        with pytest.raises(ValidationError, match="lowercase"):
            CatalogEntry(**_core_fields(work_id="Malcolm-Guite--Faith"))

    def test_work_id_special_chars_rejected(self) -> None:
        with pytest.raises(ValidationError, match="lowercase"):
            CatalogEntry(**_core_fields(work_id="malcolm_guite--faith!"))

    def test_work_id_at_max_length(self) -> None:
        author = "a" * 60
        title = "b" * 66
        work_id = f"{author}--{title}"  # 128 chars exactly
        entry = CatalogEntry(**_core_fields(work_id=work_id))
        assert len(entry.work_id) == 128


class TestSourceClassNoteValidation:
    def test_note_too_short(self) -> None:
        with pytest.raises(ValidationError, match="String should have at least 10"):
            CatalogEntry(**_core_fields(source_class_note="Short"))

    def test_note_at_minimum(self) -> None:
        entry = CatalogEntry(**_core_fields(source_class_note="1234567890"))
        assert len(entry.source_class_note) == 10


class TestGenreTagsValidation:
    def test_empty_genre_tags_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least 1"):
            CatalogEntry(**_core_fields(genre_tags=[]))


class TestSubjectHeadingsValidation:
    def test_empty_subject_headings_rejected(self) -> None:
        with pytest.raises(ValidationError, match="at least 1"):
            CatalogEntry(**_core_fields(subject_headings=[]))


class TestLanguageValidation:
    def test_iso_639_2_normalized_to_639_1(self) -> None:
        """EPUBs provide 3-char ISO 639-2 codes; validator normalizes them."""
        entry = CatalogEntry(**_core_fields(language="eng"))
        assert entry.language == "en"

    def test_iso_639_2_french(self) -> None:
        entry = CatalogEntry(**_core_fields(language="fra"))
        assert entry.language == "fr"

    def test_iso_639_2_german(self) -> None:
        entry = CatalogEntry(**_core_fields(language="deu"))
        assert entry.language == "de"

    def test_iso_639_2_bibliographic_variant(self) -> None:
        """Bibliographic variants (e.g. 'ger' for German) also normalize."""
        entry = CatalogEntry(**_core_fields(language="ger"))
        assert entry.language == "de"

    def test_unrecognized_3char_code_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ISO 639-1"):
            CatalogEntry(**_core_fields(language="zzz"))

    def test_single_char_language(self) -> None:
        with pytest.raises(ValidationError, match="ISO 639-1"):
            CatalogEntry(**_core_fields(language="e"))

    def test_four_char_code_rejected(self) -> None:
        with pytest.raises(ValidationError, match="ISO 639-1"):
            CatalogEntry(**_core_fields(language="engl"))

    def test_bcp47_tag_normalized(self) -> None:
        entry = CatalogEntry(**_core_fields(language="en-US"))
        assert entry.language == "en"

    def test_bcp47_with_639_2_prefix(self) -> None:
        """BCP-47 tag with ISO 639-2 prefix like 'eng-US' normalizes correctly."""
        entry = CatalogEntry(**_core_fields(language="eng-US"))
        assert entry.language == "en"


class TestWordCountValidation:
    def test_negative_word_count_rejected(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            CatalogEntry(**_core_fields(word_count=-1))


# ---------------------------------------------------------------------------
# PrimaryCatalogEntry tests
# ---------------------------------------------------------------------------


class TestPrimaryCatalogEntry:
    def _primary_fields(self, **overrides: object) -> dict:
        fields = _core_fields(
            source_class=SourceClass.PRIMARY,
            subject_author_id="malcolm-guite",
            work_type=WorkType.MONOGRAPH,
            voice_profile_eligible=True,
        )
        fields.update(overrides)
        return fields

    def test_valid_primary(self) -> None:
        entry = PrimaryCatalogEntry(**self._primary_fields())
        assert entry.source_class == SourceClass.PRIMARY
        assert entry.subject_author_id == "malcolm-guite"
        assert entry.work_type == WorkType.MONOGRAPH
        assert entry.voice_profile_eligible is True

    def test_optional_primary_fields(self) -> None:
        entry = PrimaryCatalogEntry(
            **self._primary_fields(
                chronological_position=2,
                dedication="For Maggie",
                table_of_contents=["Introduction", "Chapter 1"],
            )
        )
        assert entry.chronological_position == 2
        assert entry.dedication == "For Maggie"
        assert entry.table_of_contents == ["Introduction", "Chapter 1"]

    def test_wrong_source_class_rejected(self) -> None:
        with pytest.raises(ValidationError, match="source_class='primary'"):
            PrimaryCatalogEntry(**self._primary_fields(source_class=SourceClass.SECONDARY))

    def test_all_work_types(self) -> None:
        for wt in WorkType:
            entry = PrimaryCatalogEntry(**self._primary_fields(work_type=wt))
            assert entry.work_type == wt


# ---------------------------------------------------------------------------
# SecondaryCatalogEntry tests
# ---------------------------------------------------------------------------


class TestSecondaryCatalogEntry:
    def _secondary_fields(self, **overrides: object) -> dict:
        fields = _core_fields(
            source_class=SourceClass.SECONDARY,
            about_author_id="malcolm-guite",
            external_author="Holly Ordway",
            relationship=SecondaryRelationship.EDITED_COLLECTION,
            perspective_note="Sympathetic scholarly collection",
            contains_primary_quotes=True,
            quote_extraction_note="Chapter 7 by Guite",
        )
        fields.update(overrides)
        return fields

    def test_valid_secondary(self) -> None:
        entry = SecondaryCatalogEntry(**self._secondary_fields())
        assert entry.source_class == SourceClass.SECONDARY
        assert entry.about_author_id == "malcolm-guite"
        assert entry.external_author == "Holly Ordway"
        assert entry.contains_primary_quotes is True

    def test_wrong_source_class_rejected(self) -> None:
        with pytest.raises(ValidationError, match="source_class='secondary'"):
            SecondaryCatalogEntry(**self._secondary_fields(source_class=SourceClass.PRIMARY))

    def test_all_relationships(self) -> None:
        for rel in SecondaryRelationship:
            entry = SecondaryCatalogEntry(**self._secondary_fields(relationship=rel))
            assert entry.relationship == rel


# ---------------------------------------------------------------------------
# ContextualCatalogEntry tests
# ---------------------------------------------------------------------------


class TestContextualCatalogEntry:
    def _contextual_fields(self, **overrides: object) -> dict:
        fields = _core_fields(
            source_class=SourceClass.CONTEXTUAL,
            work_id="st-coleridge--biographia-literaria",
            author="Samuel Taylor Coleridge",
            referenced_by="malcolm-guite",
            engagement_type=EngagementType.INTERPRETS,
            engagement_note=(
                "Guite builds his imagination theory on "
                "Coleridge's Ch 13-14 distinction."
            ),
            engagement_works=["faith-hope-and-poetry", "mariner"],
            engagement_frequency=EngagementFrequency.FOUNDATIONAL,
        )
        fields.update(overrides)
        return fields

    def test_valid_contextual(self) -> None:
        entry = ContextualCatalogEntry(**self._contextual_fields())
        assert entry.source_class == SourceClass.CONTEXTUAL
        assert entry.referenced_by == "malcolm-guite"
        assert entry.engagement_type == EngagementType.INTERPRETS
        assert entry.engagement_frequency == EngagementFrequency.FOUNDATIONAL
        assert entry.engagement_works == ["faith-hope-and-poetry", "mariner"]

    def test_wrong_source_class_rejected(self) -> None:
        with pytest.raises(ValidationError, match="source_class='contextual'"):
            ContextualCatalogEntry(**self._contextual_fields(source_class=SourceClass.PRIMARY))


# ---------------------------------------------------------------------------
# TertiaryCatalogEntry tests
# ---------------------------------------------------------------------------


class TestTertiaryCatalogEntry:
    def _tertiary_fields(self, **overrides: object) -> dict:
        fields = _core_fields(
            source_class=SourceClass.TERTIARY,
            reference_type=ReferenceType.BIBLIOGRAPHY,
            coverage_note="Complete Guite bibliography through 2025",
        )
        fields.update(overrides)
        return fields

    def test_valid_tertiary(self) -> None:
        entry = TertiaryCatalogEntry(**self._tertiary_fields())
        assert entry.source_class == SourceClass.TERTIARY
        assert entry.reference_type == ReferenceType.BIBLIOGRAPHY

    def test_wrong_source_class_rejected(self) -> None:
        with pytest.raises(ValidationError, match="source_class='tertiary'"):
            TertiaryCatalogEntry(**self._tertiary_fields(source_class=SourceClass.PRIMARY))


# ---------------------------------------------------------------------------
# ReferenceCatalogEntry tests
# ---------------------------------------------------------------------------


class TestReferenceCatalogEntry:
    def _reference_fields(self, **overrides: object) -> dict:
        fields = _core_fields(
            source_class=SourceClass.REFERENCE,
            work_id="paul-fussell--poetic-meter-and-poetic-form",
            author="Paul Fussell",
            external_author="Paul Fussell",
            reference_type="prosody-handbook",
            subject_domain="prosody",
        )
        fields.update(overrides)
        return fields

    def test_valid_reference_has_no_voice_profile_eligibility_field(self) -> None:
        entry = ReferenceCatalogEntry(**self._reference_fields())

        assert entry.source_class == SourceClass.REFERENCE
        assert entry.external_author == "Paul Fussell"
        assert entry.reference_type == "prosody-handbook"
        assert entry.subject_domain == "prosody"
        assert "voice_profile_eligible" not in entry.model_dump()

    def test_wrong_source_class_rejected(self) -> None:
        with pytest.raises(ValidationError, match="source_class='reference'"):
            ReferenceCatalogEntry(
                **self._reference_fields(source_class=SourceClass.PRIMARY)
            )


# ---------------------------------------------------------------------------
# ClassificationResult tests
# ---------------------------------------------------------------------------


class TestClassificationResult:
    def test_valid_result(self) -> None:
        result = ClassificationResult(
            source_class=SourceClass.PRIMARY,
            confidence=0.95,
            reasoning="Guite is sole author per title page.",
            signals_detected=["authorship_match", "bibliography_match"],
        )
        assert result.source_class == SourceClass.PRIMARY
        assert result.confidence == 0.95

    def test_default_to_secondary_below_threshold(self) -> None:
        result = ClassificationResult(
            source_class=SourceClass.PRIMARY,
            confidence=0.55,
            reasoning="Uncertain classification.",
        )
        assert result.source_class == SourceClass.SECONDARY
        assert "AUTO-DOWNGRADED TO SECONDARY" in result.reasoning

    def test_secondary_stays_secondary_below_threshold(self) -> None:
        result = ClassificationResult(
            source_class=SourceClass.SECONDARY,
            confidence=0.55,
            reasoning="Already secondary.",
        )
        assert result.source_class == SourceClass.SECONDARY
        assert "AUTO-DOWNGRADED" not in result.reasoning

    def test_contextual_downgraded_below_threshold(self) -> None:
        result = ClassificationResult(
            source_class=SourceClass.CONTEXTUAL,
            confidence=0.65,
            reasoning="Might be contextual.",
        )
        assert result.source_class == SourceClass.SECONDARY

    def test_reference_downgraded_below_threshold(self) -> None:
        result = ClassificationResult(
            source_class=SourceClass.REFERENCE,
            confidence=0.65,
            reasoning="Might be a standalone reference work.",
        )
        assert result.source_class == SourceClass.SECONDARY
        assert "AUTO-DOWNGRADED TO SECONDARY" in result.reasoning

    def test_at_threshold_not_downgraded(self) -> None:
        result = ClassificationResult(
            source_class=SourceClass.PRIMARY,
            confidence=0.70,
            reasoning="Just at threshold.",
        )
        assert result.source_class == SourceClass.PRIMARY

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            ClassificationResult(
                source_class=SourceClass.PRIMARY,
                confidence=-0.1,
                reasoning="Negative",
            )
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            ClassificationResult(
                source_class=SourceClass.PRIMARY,
                confidence=1.1,
                reasoning="Over 1",
            )


# ---------------------------------------------------------------------------
# Processing route tests
# ---------------------------------------------------------------------------


class TestProcessingRoute:
    def test_primary_full_enrichment(self) -> None:
        assert route_for_source_class(SourceClass.PRIMARY) == ProcessingRoute.FULL_ENRICHMENT

    def test_secondary_embeddings_and_graph(self) -> None:
        assert route_for_source_class(SourceClass.SECONDARY) == ProcessingRoute.EMBEDDINGS_AND_GRAPH

    def test_contextual_embeddings_and_links(self) -> None:
        assert (
            route_for_source_class(SourceClass.CONTEXTUAL) == ProcessingRoute.EMBEDDINGS_AND_LINKS
        )

    def test_tertiary_metadata_only(self) -> None:
        assert route_for_source_class(SourceClass.TERTIARY) == ProcessingRoute.METADATA_ONLY

    def test_reference_enrichment(self) -> None:
        assert (
            route_for_source_class(SourceClass.REFERENCE)
            == ProcessingRoute.REFERENCE_ENRICHMENT
        )

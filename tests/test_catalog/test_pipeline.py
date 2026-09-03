"""Tests for the classification pipeline.

Tests pipeline gating logic, catalog entry construction, work_id generation,
and correct routing by source class.
"""

from __future__ import annotations

import re
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from author_library.catalog.models import (
    ClassificationResult,
    ContextualCatalogEntry,
    PrimaryCatalogEntry,
    ProcessingRoute,
    SecondaryCatalogEntry,
    SourceClass,
    TertiaryCatalogEntry,
)
from author_library.catalog.pipeline import ClassificationPipeline, PipelineResult
from author_library.config import Settings
from author_library.errors import ClassificationError
from author_library.parsing.models import (
    DocumentMetadata,
    DocumentNode,
    NodeType,
    ParsedDocument,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_document(
    *,
    title: str = "Faith, Hope and Poetry",
    author: str = "Malcolm Guite",
    raw_text: str = "Text about poetic imagination.",
    format: str = "epub",
    publisher: str | None = "Ashgate",
    publication_date: str | None = "2012",
) -> ParsedDocument:
    return ParsedDocument(
        source_path="/tmp/test.epub",
        format=format,
        metadata=DocumentMetadata(
            title=title,
            author=author,
            publisher=publisher,
            publication_date=publication_date,
            word_count=85000,
        ),
        tree=DocumentNode(node_type=NodeType.BOOK, text=raw_text),
        raw_text=raw_text,
    )


def _make_classification_result(
    source_class: SourceClass = SourceClass.PRIMARY,
    confidence: float = 0.95,
) -> ClassificationResult:
    return ClassificationResult(
        source_class=source_class,
        confidence=confidence,
        reasoning="Test classification reasoning that is long enough.",
        signals_detected=["authorship_match"],
    )


class FakeWorkRepository:
    """In-memory work repository for testing pipeline storage."""

    def __init__(self) -> None:
        self.works: dict[str, dict[str, Any]] = {}

    async def create(self, work: dict[str, Any]) -> str:
        work_id = work["work_id"]
        self.works[work_id] = work
        return work_id

    async def get(self, work_id: str) -> dict[str, Any] | None:
        return self.works.get(work_id)

    async def list_by_author(self, author: str) -> list[dict[str, Any]]:
        return [w for w in self.works.values() if w.get("author") == author]

    async def update(self, work_id: str, fields: dict[str, Any]) -> bool:
        if work_id in self.works:
            self.works[work_id].update(fields)
            return True
        return False

    async def delete(self, work_id: str) -> bool:
        return self.works.pop(work_id, None) is not None


# ---------------------------------------------------------------------------
# Work ID generation tests
# ---------------------------------------------------------------------------


class TestWorkIdGeneration:
    def test_basic_slugification(self) -> None:
        work_id = ClassificationPipeline._generate_work_id(
            "Malcolm Guite", "Faith, Hope and Poetry"
        )
        assert work_id == "malcolm-guite--faith-hope-and-poetry"

    def test_special_characters_removed(self) -> None:
        work_id = ClassificationPipeline._generate_work_id(
            "S.T. Coleridge", "Biographia Literaria: Or, Biographical Sketches"
        )
        assert "--" in work_id
        assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in work_id)

    def test_long_ids_truncated(self) -> None:
        work_id = ClassificationPipeline._generate_work_id(
            "A" * 100, "B" * 100
        )
        assert len(work_id) <= 128

    def test_work_id_matches_pattern(self) -> None:
        work_id = ClassificationPipeline._generate_work_id(
            "Malcolm Guite", "Mariner"
        )
        pattern = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*--[a-z0-9]+(?:-[a-z0-9]+)*$")
        assert pattern.match(work_id), f"work_id {work_id!r} does not match pattern"

    @pytest.mark.parametrize(
        ("author", "title"),
        [
            ("", ""),
            ("Unknown", "A Real Title"),
            ("A Real Author", "Untitled"),
        ],
    )
    def test_missing_or_sentinel_identity_raises(self, author: str, title: str) -> None:
        with pytest.raises(ValueError, match="Cannot generate work_id"):
            ClassificationPipeline._generate_work_id(author, title)

    def test_comma_format_author_normalized(self) -> None:
        """'Guite, Malcolm' and 'Malcolm Guite' must produce the same slug."""
        natural = ClassificationPipeline._generate_work_id(
            "Malcolm Guite", "Mariner"
        )
        bibliographic = ClassificationPipeline._generate_work_id(
            "Guite, Malcolm", "Mariner"
        )
        assert natural == bibliographic
        assert natural == "malcolm-guite--mariner"

    def test_comma_format_with_middle_name(self) -> None:
        """Author with middle initial in comma format."""
        work_id = ClassificationPipeline._generate_work_id(
            "Coleridge, Samuel Taylor", "The Rime of the Ancient Mariner"
        )
        assert work_id.startswith("samuel-taylor-coleridge--")

    def test_comma_format_preserves_multipart_surname(self) -> None:
        """Comma-separated author with multi-word given name."""
        work_id = ClassificationPipeline._generate_work_id(
            "von Balthasar, Hans Urs", "Prayer"
        )
        assert work_id == "hans-urs-von-balthasar--prayer"

    def test_marc_life_dates_excluded_from_slug(self) -> None:
        """MARC life dates must not leak into the work_id (2026-08-13).

        'Coleridge, Samuel Taylor, 1772-1834' produced the work_id
        samuel-taylor-1772-1834-coleridge--..., a second Work node for a work
        that already existed as samuel-taylor-coleridge--..., which split
        14,704 PART_OF edges away from the PG-matching Work node.
        """
        with_dates = ClassificationPipeline._generate_work_id(
            "Coleridge, Samuel Taylor, 1772-1834",
            "Lectures 1795 : On politics and religion",
        )
        without_dates = ClassificationPipeline._generate_work_id(
            "Samuel Taylor Coleridge",
            "Lectures 1795 : On politics and religion",
        )
        assert with_dates == without_dates
        assert with_dates.startswith("samuel-taylor-coleridge--")
        assert "1772" not in with_dates.split("--")[0]

    def test_marc_life_dates_match_existing_corpus_ids(self) -> None:
        """Re-ingesting these authors must reproduce the ids already in PG."""
        cases = [
            ("MacDonald, George, 1824-1905", "Unspoken Sermons",
             "george-macdonald--unspoken-sermons"),
            ("Nouwen, Henri J. M., 1932-1996", "Life of the Beloved",
             "henri-j-m-nouwen--life-of-the-beloved"),
        ]
        for author, title, expected in cases:
            assert ClassificationPipeline._generate_work_id(author, title) == expected


class TestNormalizeAuthorName:
    def test_natural_order_unchanged(self) -> None:
        assert ClassificationPipeline._normalize_author_name("Malcolm Guite") == "Malcolm Guite"

    def test_bibliographic_order_swapped(self) -> None:
        assert ClassificationPipeline._normalize_author_name("Guite, Malcolm") == "Malcolm Guite"

    def test_whitespace_stripped(self) -> None:
        assert (
            ClassificationPipeline._normalize_author_name("  Guite , Malcolm  ")
            == "Malcolm Guite"
        )

    def test_no_comma_no_change(self) -> None:
        assert ClassificationPipeline._normalize_author_name("Malcolm Guite") == "Malcolm Guite"

    def test_life_dates_stripped(self) -> None:
        assert (
            ClassificationPipeline._normalize_author_name(
                "Coleridge, Samuel Taylor, 1772-1834"
            )
            == "Samuel Taylor Coleridge"
        )

    def test_life_dates_variants_stripped(self) -> None:
        for raw, expected in [
            ("MacDonald, George, 1824-1905", "George MacDonald"),
            ("Nouwen, Henri J. M., 1932-1996", "Henri J. M. Nouwen"),
            ("Traherne, Thomas, 1637?-1674", "Thomas Traherne"),
            ("Someone, Jane, b. 1900", "Jane Someone"),
            ("Ancient, Author, ca. 1200", "Author Ancient"),
            ("Living, Person, 1950-", "Person Living"),
        ]:
            assert ClassificationPipeline._normalize_author_name(raw) == expected, raw

    def test_trailing_comma_still_reorders(self) -> None:
        """A MARC 100 field carries a trailing comma when $d follows.

        Regression: the life-date fix made the empty tail segment fail the
        all(parts[1:]) check, skipping the reorder entirely and slugifying the
        raw string — forking a SECOND work_id for an author that already had
        one, i.e. reintroducing the very bug it fixed.
        """
        assert (
            ClassificationPipeline._normalize_author_name("Coleridge, Samuel Taylor,")
            == "Samuel Taylor Coleridge"
        )
        assert ClassificationPipeline._generate_work_id(
            "Coleridge, Samuel Taylor,", "Some Title"
        ) == ClassificationPipeline._generate_work_id(
            "Coleridge, Samuel Taylor", "Some Title"
        )

    def test_other_marc_date_forms_stripped(self) -> None:
        """Bracketed RDA dates, approximations and centuries also fork."""
        for raw, expected in [
            ("Author, Name, [1900-1980]", "Name Author"),
            ("Author, Name, approximately 1200", "Name Author"),
            ("Author, Name, 19th cent.", "Name Author"),
            ("Author, Name, fl. 1550-1600", "Name Author"),
            ("Author, Name, d. 1674?", "Name Author"),
        ]:
            assert ClassificationPipeline._normalize_author_name(raw) == expected, raw

    def test_date_stripped_when_not_the_last_segment(self) -> None:
        """'Smith, John, 1900-1980, Jr.' — date is interior, suffix is last."""
        assert (
            ClassificationPipeline._normalize_author_name("Smith, John, 1900-1980, Jr.")
            == "John, Jr. Smith"
        )

    def test_diacritics_folded_not_deleted(self) -> None:
        """Deleting diacritics forks 'Böll' from the same author as 'Boll'."""
        assert ClassificationPipeline._generate_work_id(
            "Böll, Heinrich, 1917-1985", "Some Title"
        ) == ClassificationPipeline._generate_work_id("Boll, Heinrich", "Some Title")
        assert ClassificationPipeline._generate_work_id(
            "Böll, Heinrich", "Some Title"
        ).startswith("heinrich-boll--")

    def test_non_latin_author_does_not_collapse_to_empty(self) -> None:
        """A non-Latin author must not yield a malformed '--title' work_id.

        Every such author would otherwise collide into one id per title.
        """
        a = ClassificationPipeline._generate_work_id("Достоевский, Фёдор", "Some Title")
        b = ClassificationPipeline._generate_work_id("村上, 春樹", "Some Title")
        assert not a.startswith("--"), a
        assert not b.startswith("--"), b
        assert a != b, "distinct non-Latin authors must not collide"

    def test_surname_with_dates_only(self) -> None:
        """No given name — return the surname, not a date fragment."""
        assert (
            ClassificationPipeline._normalize_author_name("Coleridge, 1772-1834")
            == "Coleridge"
        )

    def test_non_date_qualifier_preserved(self) -> None:
        """A suffix that is not a date must keep the pre-existing behaviour."""
        assert (
            ClassificationPipeline._normalize_author_name("Smith, John, Jr.")
            == "John, Jr. Smith"
        )

    def test_empty_parts_not_swapped(self) -> None:
        """A trailing comma with no given name must not corrupt the output.

        Empty segments are now dropped, so the stray comma is removed rather
        than echoed: "Guite," normalizes to "Guite". What matters is that it
        cannot fork a second identity — assert on the work_id, which is the
        thing a difference here would actually break.
        """
        result = ClassificationPipeline._normalize_author_name("Guite,")
        assert result == "Guite"
        assert ClassificationPipeline._generate_work_id(
            "Guite,", "Mariner"
        ) == ClassificationPipeline._generate_work_id("Guite", "Mariner")


class TestMetadataFallbacks:
    def test_parseable_year_is_preserved(self) -> None:
        assert ClassificationPipeline._extract_year("2007-05-01") == 2007

    @pytest.mark.parametrize("publication_date", [None, "", "not-a-date", "????-01-01"])
    def test_unparseable_year_is_undated(self, publication_date: str | None) -> None:
        # The old current-year fallback stamped seven works with their ingestion year
        # and corrupted trace_theme chronology; an unknown year must remain unknown.
        assert ClassificationPipeline._extract_year(publication_date) is None

    @pytest.mark.parametrize(
        "publisher",
        [
            "Adobe Acrobat 9.0 Paper Capture Plug-in",
            "Internet Archive PDF 1.4.16; including mupdf and pymupdf/skimage",
            "Recoded by LuraDocument PDF v2.68",
        ],
    )
    def test_pdf_producer_is_not_a_publisher(self, publisher: str) -> None:
        assert ClassificationPipeline._clean_publisher(publisher) is None

    @pytest.mark.parametrize("publisher", [None, "", "Unknown", " unknown "])
    def test_missing_publisher_remains_unknown(self, publisher: str | None) -> None:
        assert ClassificationPipeline._clean_publisher(publisher) is None

    def test_real_publisher_is_preserved(self) -> None:
        assert ClassificationPipeline._clean_publisher(" Ashgate ") == "Ashgate"


# ---------------------------------------------------------------------------
# Pipeline routing tests
# ---------------------------------------------------------------------------


class TestPipelineProcess:
    @pytest.fixture
    def repo(self) -> FakeWorkRepository:
        return FakeWorkRepository()

    @pytest.fixture
    def settings(self) -> Settings:
        return Settings()

    def _make_pipeline(
        self, settings: Settings, repo: FakeWorkRepository
    ) -> ClassificationPipeline:
        return ClassificationPipeline(
            settings=settings,
            work_repository=repo,  # type: ignore[arg-type]
            subject_author="Malcolm Guite",
        )

    async def test_primary_route(self, settings: Settings, repo: FakeWorkRepository) -> None:
        pipeline = self._make_pipeline(settings, repo)
        classification = _make_classification_result(SourceClass.PRIMARY, 0.95)

        with patch.object(pipeline._classifier, "classify", return_value=classification):
            result = await pipeline.process(
                _make_document(),
                user_overrides={
                    "subject_author_id": "malcolm-guite",
                    "work_type": "monograph",
                    "genre_tags": ["monograph"],
                    "subject_headings": ["Imagination"],
                },
            )

        assert result.processing_route == ProcessingRoute.FULL_ENRICHMENT
        assert isinstance(result.catalog_entry, PrimaryCatalogEntry)
        assert result.catalog_entry.source_class == SourceClass.PRIMARY
        assert result.catalog_entry.work_id in repo.works

    async def test_empty_document_identity_cannot_create_work_id(
        self, settings: Settings, repo: FakeWorkRepository
    ) -> None:
        pipeline = self._make_pipeline(settings, repo)
        classification = _make_classification_result(SourceClass.PRIMARY, 0.95)

        with (
            patch.object(pipeline._classifier, "classify", return_value=classification),
            patch.object(pipeline, "_resolve_author_name", return_value=None),
            pytest.raises(ValueError, match="Cannot generate work_id"),
        ):
            await pipeline.process(_make_document(title="", author=""))

        assert repo.works == {}

    async def test_unknown_publication_metadata_stays_null(
        self, settings: Settings, repo: FakeWorkRepository
    ) -> None:
        pipeline = self._make_pipeline(settings, repo)
        classification = _make_classification_result(SourceClass.PRIMARY, 0.95)

        with patch.object(pipeline._classifier, "classify", return_value=classification):
            result = await pipeline.process(
                _make_document(
                    publisher="Adobe Acrobat 9.0 Paper Capture Plug-in",
                    publication_date="not-a-date",
                )
            )

        assert result.catalog_entry.publication_year is None
        assert result.catalog_entry.publisher is None

    async def test_secondary_route(self, settings: Settings, repo: FakeWorkRepository) -> None:
        pipeline = self._make_pipeline(settings, repo)
        classification = _make_classification_result(SourceClass.SECONDARY, 0.9)

        with patch.object(pipeline._classifier, "classify", return_value=classification):
            result = await pipeline.process(
                _make_document(author="Jane Scholar", title="Guite: A Study"),
                user_overrides={
                    "about_author_id": "malcolm-guite",
                    "external_author": "Jane Scholar",
                    "relationship": "critical-study",
                    "perspective_note": "Scholarly analysis of Guite's theology",
                    "contains_primary_quotes": False,
                    "genre_tags": ["critical-study"],
                    "subject_headings": ["Guite criticism"],
                },
            )

        assert result.processing_route == ProcessingRoute.EMBEDDINGS_AND_GRAPH
        assert isinstance(result.catalog_entry, SecondaryCatalogEntry)

    async def test_contextual_route(self, settings: Settings, repo: FakeWorkRepository) -> None:
        pipeline = self._make_pipeline(settings, repo)
        classification = _make_classification_result(SourceClass.CONTEXTUAL, 0.85)

        with patch.object(pipeline._classifier, "classify", return_value=classification):
            result = await pipeline.process(
                _make_document(
                    author="Samuel Taylor Coleridge",
                    title="Biographia Literaria",
                ),
                user_overrides={
                    "referenced_by": "malcolm-guite",
                    "engagement_type": "interprets",
                    "engagement_note": "Foundation for Guite's imagination theory.",
                    "genre_tags": ["philosophy"],
                    "subject_headings": ["Imagination"],
                },
            )

        assert result.processing_route == ProcessingRoute.EMBEDDINGS_AND_LINKS
        assert isinstance(result.catalog_entry, ContextualCatalogEntry)

    async def test_tertiary_route(self, settings: Settings, repo: FakeWorkRepository) -> None:
        pipeline = self._make_pipeline(settings, repo)
        classification = _make_classification_result(SourceClass.TERTIARY, 0.9)

        with patch.object(pipeline._classifier, "classify", return_value=classification):
            result = await pipeline.process(
                _make_document(
                    author="Bibliographer",
                    title="Complete Guite Bibliography",
                ),
                user_overrides={
                    "reference_type": "bibliography",
                    "coverage_note": "Complete through 2025",
                    "genre_tags": ["bibliography"],
                    "subject_headings": ["Guite bibliography"],
                },
            )

        assert result.processing_route == ProcessingRoute.METADATA_ONLY
        assert isinstance(result.catalog_entry, TertiaryCatalogEntry)

    async def test_entry_stored_in_repository(
        self, settings: Settings, repo: FakeWorkRepository
    ) -> None:
        pipeline = self._make_pipeline(settings, repo)
        classification = _make_classification_result(SourceClass.PRIMARY, 0.95)

        with patch.object(pipeline._classifier, "classify", return_value=classification):
            result = await pipeline.process(
                _make_document(),
                user_overrides={
                    "genre_tags": ["monograph"],
                    "subject_headings": ["Imagination"],
                },
            )

        # Verify work was stored
        stored = await repo.get(result.catalog_entry.work_id)
        assert stored is not None
        assert stored["title"] == "Faith, Hope and Poetry"
        assert stored["source_class"] == "primary"

    async def test_user_overrides_applied(
        self, settings: Settings, repo: FakeWorkRepository
    ) -> None:
        pipeline = self._make_pipeline(settings, repo)
        classification = _make_classification_result(SourceClass.PRIMARY, 0.95)

        with patch.object(pipeline._classifier, "classify", return_value=classification):
            result = await pipeline.process(
                _make_document(),
                user_overrides={
                    "work_id": "malcolm-guite--custom-id",
                    "publication_year": 2010,
                    "genre_tags": ["poetry-collection"],
                    "subject_headings": ["Poetry"],
                    "subject_author_id": "malcolm-guite",
                    "work_type": "poetry-collection",
                },
            )

        assert result.catalog_entry.work_id == "malcolm-guite--custom-id"

    async def test_storage_failure_raises(
        self, settings: Settings, repo: FakeWorkRepository
    ) -> None:
        pipeline = self._make_pipeline(settings, repo)
        classification = _make_classification_result(SourceClass.PRIMARY, 0.95)

        # Make repository fail
        repo.create = AsyncMock(side_effect=RuntimeError("DB connection lost"))  # type: ignore[method-assign]

        with (
            patch.object(pipeline._classifier, "classify", return_value=classification),
            pytest.raises(ClassificationError, match="Failed to store"),
        ):
            await pipeline.process(
                _make_document(),
                user_overrides={
                    "genre_tags": ["monograph"],
                    "subject_headings": ["Imagination"],
                },
            )


# ---------------------------------------------------------------------------
# Pipeline result tests
# ---------------------------------------------------------------------------


class TestPipelineResult:
    def test_repr(self) -> None:
        entry = PrimaryCatalogEntry(
            work_id="malcolm-guite--test",
            title="Test",
            author="Malcolm Guite",
            source_class=SourceClass.PRIMARY,
            source_class_note="Test classification note for testing.",
            publication_year=2020,
            publisher="Test Press",
            format_ingested="epub",
            word_count=1000,
            genre_tags=["monograph"],
            subject_headings=["Test"],
            subject_author_id="malcolm-guite",
            work_type="monograph",
        )
        classification = _make_classification_result()
        result = PipelineResult(
            catalog_entry=entry,
            classification=classification,
            processing_route=ProcessingRoute.FULL_ENRICHMENT,
        )
        repr_str = repr(result)
        assert "malcolm-guite--test" in repr_str
        assert "full_enrichment" in repr_str

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
) -> ParsedDocument:
    return ParsedDocument(
        source_path="/tmp/test.epub",
        format=format,
        metadata=DocumentMetadata(
            title=title,
            author=author,
            publisher="Ashgate",
            publication_date="2012",
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


class TestNormalizeAuthorName:
    def test_natural_order_unchanged(self) -> None:
        assert ClassificationPipeline._normalize_author_name("Malcolm Guite") == "Malcolm Guite"

    def test_bibliographic_order_swapped(self) -> None:
        assert ClassificationPipeline._normalize_author_name("Guite, Malcolm") == "Malcolm Guite"

    def test_whitespace_stripped(self) -> None:
        assert ClassificationPipeline._normalize_author_name("  Guite , Malcolm  ") == "Malcolm Guite"

    def test_no_comma_no_change(self) -> None:
        assert ClassificationPipeline._normalize_author_name("Malcolm Guite") == "Malcolm Guite"

    def test_empty_parts_not_swapped(self) -> None:
        """A trailing comma with no given name should not corrupt the output."""
        # "Guite," has an empty second part — should not swap
        result = ClassificationPipeline._normalize_author_name("Guite,")
        assert result.strip() == "Guite,"


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

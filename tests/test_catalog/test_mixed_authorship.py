"""Tests for mixed-authorship detection and handling.

Tests edited collection chapter detection, interview Q&A splitting,
foreword/afterword detection, and foreign segment detection in primary works.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from author_library.catalog.mixed_authorship import (
    AuthorshipSegment,
    MixedAuthorshipAnalyzer,
    MixedAuthorshipResult,
    SegmentType,
)
from author_library.catalog.models import SourceClass
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
    title: str = "Test Document",
    author: str = "Editor Name",
    tree: DocumentNode | None = None,
) -> ParsedDocument:
    if tree is None:
        tree = DocumentNode(node_type=NodeType.BOOK, text="Default text")
    return ParsedDocument(
        source_path="/tmp/test.epub",
        format="epub",
        metadata=DocumentMetadata(
            title=title,
            author=author,
            word_count=50000,
        ),
        tree=tree,
        raw_text="",
    )


def _make_chapter(
    *,
    title: str = "Chapter 1",
    text: str = "Chapter text",
    author: str = "",
    children: list[DocumentNode] | None = None,
) -> DocumentNode:
    metadata: dict[str, str | int | bool | list[str]] = {}
    if title:
        metadata["title"] = title
    if author:
        metadata["author"] = author
    return DocumentNode(
        node_type=NodeType.CHAPTER,
        text=text,
        metadata=metadata,
        children=children or [],
    )


# ---------------------------------------------------------------------------
# Basic analyzer tests
# ---------------------------------------------------------------------------


class TestMixedAuthorshipAnalyzer:
    @pytest.fixture
    def analyzer(self) -> MixedAuthorshipAnalyzer:
        return MixedAuthorshipAnalyzer(subject_author="Malcolm Guite")

    def test_no_mixed_content_in_simple_primary(
        self, analyzer: MixedAuthorshipAnalyzer
    ) -> None:
        doc = _make_document(
            title="Faith, Hope and Poetry",
            author="Malcolm Guite",
            tree=DocumentNode(
                node_type=NodeType.BOOK,
                children=[
                    _make_chapter(title="Introduction", author="Malcolm Guite"),
                    _make_chapter(title="Chapter 1", author="Malcolm Guite"),
                ],
            ),
        )
        result = analyzer.analyze(doc, document_source_class=SourceClass.PRIMARY)
        assert not result.is_mixed

    def test_no_mixed_content_in_simple_secondary(
        self, analyzer: MixedAuthorshipAnalyzer
    ) -> None:
        doc = _make_document(
            title="A Study of Modern Poetry",
            author="Jane Scholar",
            tree=DocumentNode(
                node_type=NodeType.BOOK,
                children=[
                    _make_chapter(title="Introduction", author="Jane Scholar"),
                    _make_chapter(title="Chapter 1", author="Jane Scholar"),
                ],
            ),
        )
        result = analyzer.analyze(doc, document_source_class=SourceClass.SECONDARY)
        assert not result.is_mixed


# ---------------------------------------------------------------------------
# Edited collection tests (subject author as contributor)
# ---------------------------------------------------------------------------


class TestEditedCollectionDetection:
    @pytest.fixture
    def analyzer(self) -> MixedAuthorshipAnalyzer:
        return MixedAuthorshipAnalyzer(subject_author="Malcolm Guite")

    def test_detect_subject_author_chapter_by_metadata(
        self, analyzer: MixedAuthorshipAnalyzer
    ) -> None:
        """Per classification-examples.md §2: subject author chapter in edited collection."""
        doc = _make_document(
            title="Tolkien's Modern Middle Ages",
            author="Holly Ordway (ed.)",
            tree=DocumentNode(
                node_type=NodeType.BOOK,
                children=[
                    _make_chapter(title="Introduction", author="Holly Ordway"),
                    _make_chapter(title="Tolkien and Barfield", author="Another Scholar"),
                    _make_chapter(
                        title="Tolkien's Poetic Imagination",
                        author="Malcolm Guite",
                        text=(
                            "In this chapter I argue that Tolkien's "
                            "understanding of imagination..."
                        ),
                    ),
                    _make_chapter(title="Tolkien and Lewis", author="Third Scholar"),
                ],
            ),
        )

        result = analyzer.analyze(doc, document_source_class=SourceClass.SECONDARY)

        assert result.is_mixed
        assert result.primary_adjacent_count == 1
        assert result.requires_extraction

        # Find the primary-adjacent segment
        primary_segs = [s for s in result.segments if s.is_primary_adjacent]
        assert len(primary_segs) == 1
        assert primary_segs[0].attributed_author == "Malcolm Guite"
        assert primary_segs[0].source_class == SourceClass.PRIMARY
        assert primary_segs[0].segment_type == SegmentType.CHAPTER

    def test_detect_subject_author_chapter_by_title(
        self, analyzer: MixedAuthorshipAnalyzer
    ) -> None:
        doc = _make_document(
            title="Essays on Imagination",
            author="Various",
            tree=DocumentNode(
                node_type=NodeType.BOOK,
                children=[
                    _make_chapter(title="by Malcolm Guite: On Poetry"),
                ],
            ),
        )

        result = analyzer.analyze(doc, document_source_class=SourceClass.SECONDARY)
        assert result.is_mixed
        assert result.primary_adjacent_count >= 1

    def test_no_false_positive_for_unrelated_chapters(
        self, analyzer: MixedAuthorshipAnalyzer
    ) -> None:
        doc = _make_document(
            title="Collected Essays",
            author="Editor",
            tree=DocumentNode(
                node_type=NodeType.BOOK,
                children=[
                    _make_chapter(title="Chapter 1", author="Someone Else"),
                    _make_chapter(title="Chapter 2", author="Another Person"),
                ],
            ),
        )

        result = analyzer.analyze(doc, document_source_class=SourceClass.SECONDARY)
        assert not result.is_mixed


# ---------------------------------------------------------------------------
# Interview detection tests
# ---------------------------------------------------------------------------


class TestInterviewDetection:
    @pytest.fixture
    def analyzer(self) -> MixedAuthorshipAnalyzer:
        return MixedAuthorshipAnalyzer(subject_author="Malcolm Guite")

    def test_detect_interview_by_title(
        self, analyzer: MixedAuthorshipAnalyzer
    ) -> None:
        """Per classification-examples.md §3: Interview Q&A splitting."""
        doc = _make_document(
            title="A Conversation with Malcolm Guite",
            author="Literary Magazine",
            tree=DocumentNode(node_type=NodeType.BOOK, text="Q: What is poetry? A: Poetry is..."),
        )

        result = analyzer.analyze(doc, document_source_class=SourceClass.SECONDARY)

        assert result.is_mixed
        assert result.primary_adjacent_count >= 1

        # Should have both interview response (primary) and question (secondary) segments
        types = {s.segment_type for s in result.segments}
        assert SegmentType.INTERVIEW_RESPONSE in types
        assert SegmentType.INTERVIEW_QUESTION in types

    def test_detect_interview_q_and_a(
        self, analyzer: MixedAuthorshipAnalyzer
    ) -> None:
        doc = _make_document(
            title="Q&A with Malcolm Guite on Poetry",
            author="Interviewer",
        )

        result = analyzer.analyze(doc, document_source_class=SourceClass.SECONDARY)
        assert result.is_mixed

    def test_no_interview_for_non_interview_title(
        self, analyzer: MixedAuthorshipAnalyzer
    ) -> None:
        doc = _make_document(
            title="A Critical Study of Modern Poetry",
            author="Scholar",
        )

        result = analyzer.analyze(doc, document_source_class=SourceClass.SECONDARY)
        # No interview segments should be created (may still have other mixed content)
        interview_segs = [
            s
            for s in result.segments
            if s.segment_type
            in (SegmentType.INTERVIEW_QUESTION, SegmentType.INTERVIEW_RESPONSE)
        ]
        assert len(interview_segs) == 0


# ---------------------------------------------------------------------------
# Foreword detection in primary works
# ---------------------------------------------------------------------------


class TestForeignSegmentsInPrimary:
    @pytest.fixture
    def analyzer(self) -> MixedAuthorshipAnalyzer:
        return MixedAuthorshipAnalyzer(subject_author="Malcolm Guite")

    def test_detect_foreign_foreword(
        self, analyzer: MixedAuthorshipAnalyzer
    ) -> None:
        """Per classification-examples.md §4: Foreign foreword in primary work."""
        doc = _make_document(
            title="Faith, Hope and Poetry",
            author="Malcolm Guite",
            tree=DocumentNode(
                node_type=NodeType.BOOK,
                children=[
                    _make_chapter(
                        title="Foreword",
                        author="Rowan Williams",
                        text="Guite's work represents a major contribution...",
                    ),
                    _make_chapter(title="Introduction", author="Malcolm Guite"),
                    _make_chapter(title="Chapter 1", author="Malcolm Guite"),
                ],
            ),
        )

        result = analyzer.analyze(doc, document_source_class=SourceClass.PRIMARY)

        assert result.is_mixed
        secondary_segs = [s for s in result.segments if not s.is_primary_adjacent]
        assert len(secondary_segs) == 1
        assert secondary_segs[0].attributed_author == "Rowan Williams"
        assert secondary_segs[0].source_class == SourceClass.SECONDARY
        assert secondary_segs[0].segment_type == SegmentType.FOREWORD

    def test_no_false_positive_for_author_foreword(
        self, analyzer: MixedAuthorshipAnalyzer
    ) -> None:
        """Author's own foreword should not be flagged as foreign."""
        doc = _make_document(
            title="Faith, Hope and Poetry",
            author="Malcolm Guite",
            tree=DocumentNode(
                node_type=NodeType.BOOK,
                children=[
                    _make_chapter(
                        title="Foreword",
                        author="Malcolm Guite",
                        text="In writing this foreword I want to...",
                    ),
                    _make_chapter(title="Chapter 1", author="Malcolm Guite"),
                ],
            ),
        )

        result = analyzer.analyze(doc, document_source_class=SourceClass.PRIMARY)
        # Author's own foreword should not create a segment
        assert not result.is_mixed


# ---------------------------------------------------------------------------
# Contextual source tests
# ---------------------------------------------------------------------------


class TestContextualSourceHandling:
    @pytest.fixture
    def analyzer(self) -> MixedAuthorshipAnalyzer:
        return MixedAuthorshipAnalyzer(subject_author="Malcolm Guite")

    def test_contextual_not_analyzed_for_mixed(
        self, analyzer: MixedAuthorshipAnalyzer
    ) -> None:
        """Contextual sources don't typically contain subject-author content."""
        doc = _make_document(
            title="Biographia Literaria",
            author="Samuel Taylor Coleridge",
        )

        result = analyzer.analyze(doc, document_source_class=SourceClass.CONTEXTUAL)
        assert not result.is_mixed


# ---------------------------------------------------------------------------
# Analysis notes tests
# ---------------------------------------------------------------------------


class TestAnalysisNotes:
    @pytest.fixture
    def analyzer(self) -> MixedAuthorshipAnalyzer:
        return MixedAuthorshipAnalyzer(subject_author="Malcolm Guite")

    def test_notes_for_no_mixed_content(
        self, analyzer: MixedAuthorshipAnalyzer
    ) -> None:
        doc = _make_document(title="Simple Book", author="Author")
        result = analyzer.analyze(doc, document_source_class=SourceClass.PRIMARY)
        assert "No mixed-authorship detected" in result.analysis_notes

    def test_notes_for_mixed_content(
        self, analyzer: MixedAuthorshipAnalyzer
    ) -> None:
        doc = _make_document(
            title="Edited Collection",
            author="Editor",
            tree=DocumentNode(
                node_type=NodeType.BOOK,
                children=[
                    _make_chapter(
                        title="Guite's Chapter",
                        author="Malcolm Guite",
                        text="My argument about imagination...",
                    ),
                ],
            ),
        )
        result = analyzer.analyze(doc, document_source_class=SourceClass.SECONDARY)
        assert "Mixed-authorship detected" in result.analysis_notes
        assert "primary-adjacent" in result.analysis_notes


# ---------------------------------------------------------------------------
# AuthorshipSegment model tests
# ---------------------------------------------------------------------------


class TestAuthorshipSegment:
    def test_segment_creation(self) -> None:
        seg = AuthorshipSegment(
            segment_type=SegmentType.CHAPTER,
            title="Chapter 7",
            attributed_author="Malcolm Guite",
            source_class=SourceClass.PRIMARY,
            is_primary_adjacent=True,
            confidence=0.9,
            notes="Subject author chapter in edited collection.",
        )
        assert seg.segment_type == SegmentType.CHAPTER
        assert seg.is_primary_adjacent
        assert seg.confidence == 0.9

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            AuthorshipSegment(
                segment_type=SegmentType.CHAPTER,
                title="Test",
                attributed_author="Test",
                source_class=SourceClass.PRIMARY,
                confidence=1.5,
            )


class TestMixedAuthorshipResult:
    def test_empty_result(self) -> None:
        result = MixedAuthorshipResult()
        assert not result.is_mixed
        assert result.segments == []
        assert result.primary_adjacent_count == 0
        assert not result.requires_extraction

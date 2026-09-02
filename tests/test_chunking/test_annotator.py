"""Tests for the contextual annotation engine."""

from __future__ import annotations

import os

import pytest

from author_library.chunking.annotator import (
    AnnotationContext,
    ChunkAnnotator,
    _publication_year_label,
)
from author_library.chunking.models import Chunk, ChunkGranularity


@pytest.fixture
def primary_context() -> AnnotationContext:
    return AnnotationContext(
        work_title="Faith, Hope and Poetry",
        publication_year=2010,
        author="Malcolm Guite",
        subject_author="Malcolm Guite",
        chapter_title="The Poetic Imagination",
        chapter_number=1,
    )


@pytest.fixture
def secondary_context() -> AnnotationContext:
    return AnnotationContext(
        work_title="The Art of Malcolm Guite",
        publication_year=2020,
        author="Jane Doe",
        subject_author="Malcolm Guite",
        relationship_type="critical study",
        perspective_note="Focuses on Guite's theological aesthetics",
    )


@pytest.fixture
def contextual_context() -> AnnotationContext:
    return AnnotationContext(
        work_title="Biographia Literaria",
        publication_year=1817,
        author="Samuel Taylor Coleridge",
        subject_author="Malcolm Guite",
        engagement_note="Foundational text for Guite's theory of imagination",
        engagement_works="Faith, Hope and Poetry; Mariner",
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            text=(
                "Coleridge's distinction between Primary and Secondary "
                "Imagination forms the cornerstone of his poetic philosophy."
            ),
            granularity=ChunkGranularity.MESO,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
            chapter="The Poetic Imagination",
            position=0,
        ),
        Chunk(
            text="The implications for theology are profound.",
            granularity=ChunkGranularity.MICRO,
            work_id="guite--faith-hope-poetry",
            source_class="primary",
            chapter="The Poetic Imagination",
            position=1,
        ),
    ]


class TestAnnotationTemplates:
    """Test annotation templates without LLM (template-only mode)."""

    async def test_primary_annotation_structure(
        self, sample_chunks: list[Chunk], primary_context: AnnotationContext
    ) -> None:
        annotator = ChunkAnnotator()
        result = await annotator.annotate_chunks(sample_chunks, primary_context)
        for chunk in result:
            assert chunk.annotation is not None
            assert "[PRIMARY]" in chunk.annotation
            assert "Faith, Hope and Poetry" in chunk.annotation
            assert "Malcolm Guite" in chunk.annotation

    async def test_secondary_annotation_structure(
        self, secondary_context: AnnotationContext
    ) -> None:
        chunks = [
            Chunk(
                text="Guite's theology of imagination draws heavily on Coleridge.",
                granularity=ChunkGranularity.MESO,
                work_id="doe--art-of-guite",
                source_class="secondary",
                position=0,
            ),
        ]
        annotator = ChunkAnnotator()
        result = await annotator.annotate_chunks(chunks, secondary_context)
        assert result[0].annotation is not None
        assert "[SECONDARY:" in result[0].annotation
        assert "Jane Doe" in result[0].annotation
        assert "Malcolm Guite" in result[0].annotation
        assert "critical study" in result[0].annotation

    async def test_contextual_annotation_structure(
        self, contextual_context: AnnotationContext
    ) -> None:
        chunks = [
            Chunk(
                text="The primary IMAGINATION I hold to be the living Power.",
                granularity=ChunkGranularity.MESO,
                work_id="coleridge--biographia-literaria",
                source_class="contextual",
                position=0,
            ),
        ]
        annotator = ChunkAnnotator()
        result = await annotator.annotate_chunks(chunks, contextual_context)
        assert result[0].annotation is not None
        assert "[CONTEXTUAL:" in result[0].annotation
        assert "Coleridge" in result[0].annotation
        assert "Malcolm Guite" in result[0].annotation
        assert "Foundational text" in result[0].annotation

    async def test_source_class_markers_present(
        self, sample_chunks: list[Chunk], primary_context: AnnotationContext
    ) -> None:
        """Source classification markers MUST be present to prevent voice contamination."""
        annotator = ChunkAnnotator()
        result = await annotator.annotate_chunks(sample_chunks, primary_context)
        for chunk in result:
            assert chunk.annotation is not None
            assert "[PRIMARY]" in chunk.annotation

    async def test_empty_chunks_list(self, primary_context: AnnotationContext) -> None:
        annotator = ChunkAnnotator()
        result = await annotator.annotate_chunks([], primary_context)
        assert result == []

    def test_unknown_publication_year_is_labeled_undated(self) -> None:
        assert _publication_year_label(None) == "undated"

    async def test_annotation_includes_chapter_info(
        self, sample_chunks: list[Chunk], primary_context: AnnotationContext
    ) -> None:
        annotator = ChunkAnnotator()
        result = await annotator.annotate_chunks(sample_chunks, primary_context)
        assert any("Chapter 1" in (c.annotation or "") for c in result)

    async def test_annotated_text_property_after_annotation(
        self, sample_chunks: list[Chunk], primary_context: AnnotationContext
    ) -> None:
        annotator = ChunkAnnotator()
        result = await annotator.annotate_chunks(sample_chunks, primary_context)
        for chunk in result:
            assert chunk.annotation is not None
            assert chunk.annotated_text.startswith("[PRIMARY]")
            assert chunk.text in chunk.annotated_text


class TestAnnotationWithLLM:
    """Tests that require the Anthropic API key."""

    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    async def test_llm_enriched_annotation(
        self, sample_chunks: list[Chunk], primary_context: AnnotationContext
    ) -> None:
        annotator = ChunkAnnotator()
        result = await annotator.annotate_chunks(sample_chunks, primary_context)
        for chunk in result:
            assert chunk.annotation is not None
            # LLM should generate a topic description
            assert "covers:" in chunk.annotation

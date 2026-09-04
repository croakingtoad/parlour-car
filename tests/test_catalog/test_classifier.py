"""Tests for the source classification engine.

Tests the classification engine's response parsing, error handling,
and the safety rule. LLM API tests are skipped if no API key is set.
"""

from __future__ import annotations

import json
import os

import pytest

from author_library.catalog.classifier import (
    CLASSIFICATION_SYSTEM_PROMPT,
    SourceClassifier,
    _build_classification_prompt,
    _extract_text_from_tree,
)
from author_library.catalog.models import SourceClass
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

HAS_ANTHROPIC_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))


def _make_document(
    *,
    title: str = "Faith, Hope and Poetry",
    author: str = "Malcolm Guite",
    raw_text: str = "This is a scholarly monograph exploring the theology of poetic imagination.",
    publisher: str | None = "Ashgate",
    publication_date: str | None = "2012",
) -> ParsedDocument:
    return ParsedDocument(
        source_path="/tmp/test.epub",
        format="epub",
        metadata=DocumentMetadata(
            title=title,
            author=author,
            publisher=publisher,
            publication_date=publication_date,
            word_count=85000,
        ),
        tree=DocumentNode(
            node_type=NodeType.BOOK,
            text=raw_text,
        ),
        raw_text=raw_text,
    )


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------


class TestBuildClassificationPrompt:
    def test_taxonomy_distinguishes_reference_contextual_and_tertiary(self) -> None:
        assert "REFERENCE" in CLASSIFICATION_SYSTEM_PROMPT
        assert '"reference"' in CLASSIFICATION_SYSTEM_PROMPT
        assert "evidence" in CLASSIFICATION_SYSTEM_PROMPT.lower()
        assert "no relationship" in CLASSIFICATION_SYSTEM_PROMPT.lower()
        assert "catalogue-only" in CLASSIFICATION_SYSTEM_PROMPT.lower()

    def test_includes_subject_author(self) -> None:
        prompt = _build_classification_prompt(
            subject_author="Malcolm Guite",
            document_author="Malcolm Guite",
            document_title="Faith, Hope and Poetry",
            publisher="Ashgate",
            publication_year=2012,
            content_sample="Some text about imagination.",
            metadata_hints=None,
        )
        assert "Malcolm Guite" in prompt
        assert "Faith, Hope and Poetry" in prompt
        assert "Ashgate" in prompt
        assert "2012" in prompt

    def test_includes_metadata_hints(self) -> None:
        prompt = _build_classification_prompt(
            subject_author="Malcolm Guite",
            document_author=None,
            document_title=None,
            publisher=None,
            publication_year=None,
            content_sample="text",
            metadata_hints={"known_bibliography": "true", "genre": "monograph"},
        )
        assert "known_bibliography" in prompt
        assert "monograph" in prompt

    def test_content_sample_truncated(self) -> None:
        long_sample = "word " * 2000  # 10000 chars
        prompt = _build_classification_prompt(
            subject_author="Test",
            document_author=None,
            document_title=None,
            publisher=None,
            publication_year=None,
            content_sample=long_sample,
            metadata_hints=None,
        )
        # Content sample is truncated to 4000 chars
        assert len(prompt) < len(long_sample) + 500


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def _get_classifier(self) -> SourceClassifier:
        settings = Settings()
        return SourceClassifier(settings)

    def test_parse_valid_json(self) -> None:
        classifier = self._get_classifier()
        response = json.dumps({
            "source_class": "primary",
            "confidence": 0.95,
            "reasoning": "Author matches subject.",
            "signals_detected": ["authorship_match"],
        })
        result = classifier._parse_classification_response(response)
        assert result.source_class == SourceClass.PRIMARY
        assert result.confidence == 0.95

    def test_parse_json_with_code_fences(self) -> None:
        classifier = self._get_classifier()
        response = (
            '```json\n'
            '{"source_class": "secondary", "confidence": 0.8, '
            '"reasoning": "Different author.", "signals_detected": []}'
            '\n```'
        )
        result = classifier._parse_classification_response(response)
        assert result.source_class == SourceClass.SECONDARY

    def test_parse_invalid_json_raises(self) -> None:
        classifier = self._get_classifier()
        with pytest.raises(ClassificationError, match="Failed to parse"):
            classifier._parse_classification_response("not json at all")

    def test_parse_missing_fields_raises(self) -> None:
        classifier = self._get_classifier()
        response = json.dumps({"source_class": "primary"})
        with pytest.raises(ClassificationError, match="Invalid classification response"):
            classifier._parse_classification_response(response)

    def test_parse_invalid_source_class_raises(self) -> None:
        classifier = self._get_classifier()
        response = json.dumps({
            "source_class": "unknown",
            "confidence": 0.5,
            "reasoning": "Bad class.",
        })
        with pytest.raises(ClassificationError, match="Invalid classification response"):
            classifier._parse_classification_response(response)

    def test_low_confidence_triggers_secondary_default(self) -> None:
        classifier = self._get_classifier()
        response = json.dumps({
            "source_class": "primary",
            "confidence": 0.5,
            "reasoning": "Uncertain.",
            "signals_detected": [],
        })
        result = classifier._parse_classification_response(response)
        # The ClassificationResult model applies the default-to-secondary rule
        assert result.source_class == SourceClass.SECONDARY
        assert "AUTO-DOWNGRADED" in result.reasoning

    def test_parse_reference_classification(self) -> None:
        classifier = self._get_classifier()
        response = json.dumps({
            "source_class": "reference",
            "confidence": 0.93,
            "reasoning": "Standalone prosody handbook with no subject-author relationship.",
            "signals_detected": ["third_party_authorship", "no_author_relationship"],
        })
        result = classifier._parse_classification_response(response)
        assert result.source_class == SourceClass.REFERENCE

    def test_low_confidence_reference_triggers_secondary_default(self) -> None:
        classifier = self._get_classifier()
        response = json.dumps({
            "source_class": "reference",
            "confidence": 0.6,
            "reasoning": "Relationship is uncertain.",
            "signals_detected": [],
        })
        result = classifier._parse_classification_response(response)
        assert result.source_class == SourceClass.SECONDARY
        assert "AUTO-DOWNGRADED" in result.reasoning


# ---------------------------------------------------------------------------
# Content extraction tests
# ---------------------------------------------------------------------------


class TestContentExtraction:
    def test_extract_from_raw_text(self) -> None:
        doc = _make_document(raw_text="word " * 2000)
        classifier = SourceClassifier(Settings())
        sample = classifier._extract_content_sample(doc)
        words = sample.split()
        assert len(words) == 1000

    def test_extract_from_tree_fallback(self) -> None:
        doc = ParsedDocument(
            source_path="/tmp/test.epub",
            format="epub",
            metadata=DocumentMetadata(title="Test", word_count=0),
            tree=DocumentNode(
                node_type=NodeType.BOOK,
                children=[
                    DocumentNode(node_type=NodeType.PARAGRAPH, text="Hello world from tree."),
                ],
            ),
            raw_text="",
        )
        classifier = SourceClassifier(Settings())
        sample = classifier._extract_content_sample(doc)
        assert "Hello world from tree" in sample

    def test_extract_text_from_nested_tree(self) -> None:
        root = DocumentNode(
            node_type=NodeType.BOOK,
            text="Root",
            children=[
                DocumentNode(
                    node_type=NodeType.CHAPTER,
                    text="Chapter 1",
                    children=[
                        DocumentNode(node_type=NodeType.PARAGRAPH, text="Paragraph A."),
                        DocumentNode(node_type=NodeType.PARAGRAPH, text="Paragraph B."),
                    ],
                ),
            ],
        )
        text = _extract_text_from_tree(root)
        assert "Root" in text
        assert "Chapter 1" in text
        assert "Paragraph A" in text
        assert "Paragraph B" in text


# ---------------------------------------------------------------------------
# LLM integration tests (skipped without API key)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
class TestLLMClassification:
    """Integration tests that call the real Anthropic API."""

    async def test_classify_primary_source(self) -> None:
        settings = Settings()
        classifier = SourceClassifier(settings)
        doc = _make_document(
            title="Faith, Hope and Poetry: Theology and the Poetic Imagination",
            author="Malcolm Guite",
            raw_text=(
                "In this book I argue that the poetic imagination is not merely "
                "decorative but a truth-bearing faculty. I trace this argument through "
                "the English literary tradition from Caedmon to Seamus Heaney, showing "
                "how poets have understood imagination as a gateway to transcendent truth."
            ),
        )
        result = await classifier.classify(
            doc,
            subject_author="Malcolm Guite",
        )
        assert result.source_class == SourceClass.PRIMARY
        assert result.confidence >= 0.7

    async def test_classify_secondary_source(self) -> None:
        settings = Settings()
        classifier = SourceClassifier(settings)
        doc = _make_document(
            title="Malcolm Guite: A Critical Introduction",
            author="Jane Scholar",
            raw_text=(
                "Malcolm Guite's contribution to the theology of imagination represents "
                "a significant development in contemporary Anglican thought. In this study, "
                "we examine his major works and trace the intellectual lineage of his arguments "
                "about poetry as a truth-bearing faculty."
            ),
        )
        result = await classifier.classify(
            doc,
            subject_author="Malcolm Guite",
        )
        assert result.source_class == SourceClass.SECONDARY

    async def test_classify_contextual_source(self) -> None:
        settings = Settings()
        classifier = SourceClassifier(settings)
        doc = _make_document(
            title="Biographia Literaria",
            author="Samuel Taylor Coleridge",
            raw_text=(
                "The IMAGINATION then I consider either as primary, or secondary. "
                "The primary IMAGINATION I hold to be the living Power and prime Agent "
                "of all human Perception, and as a repetition in the finite mind of "
                "the eternal act of creation in the infinite I AM."
            ),
            publisher="Various",
            publication_date="1817",
        )
        result = await classifier.classify(
            doc,
            subject_author="Malcolm Guite",
            metadata_hints={
                "known_engagement": (
                    "Guite extensively references this work in "
                    "Faith, Hope and Poetry and Mariner"
                ),
            },
        )
        # Contextual or secondary both acceptable — the key is it's not primary
        assert result.source_class in (SourceClass.CONTEXTUAL, SourceClass.SECONDARY)

    async def test_no_api_key_raises(self) -> None:
        settings = Settings()
        # Override the API key to be empty
        settings.api_keys.anthropic_api_key = type(settings.api_keys.anthropic_api_key)("")
        classifier = SourceClassifier(settings)
        doc = _make_document()
        with pytest.raises(ClassificationError, match="API key is required"):
            await classifier.classify(doc, subject_author="Malcolm Guite")

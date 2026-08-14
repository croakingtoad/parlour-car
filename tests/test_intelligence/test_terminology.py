"""Tests for terminology normalization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from author_library.intelligence.terminology import (
    AmbiguousTerm,
    TerminologyMap,
    TerminologyNormalizer,
    TermMapping,
    _extract_terms_from_chunks,
)

if TYPE_CHECKING:
    from author_library.config import Settings

from tests.test_intelligence.conftest import (
    PRIMARY_CHUNKS,
    requires_anthropic_key,
)

# ---------------------------------------------------------------------------
# Model validation tests
# ---------------------------------------------------------------------------


class TestTerminologyModels:
    """Test terminology data model validation."""

    def test_term_mapping_creation(self) -> None:
        """A term mapping should validate."""
        mapping = TermMapping(
            variant="imaginative faculty",
            canonical="Imagination",
            relationship="synonym",
            confidence=0.95,
            notes="Coleridge uses both interchangeably",
        )
        assert mapping.canonical == "Imagination"
        assert mapping.confidence == 0.95

    def test_ambiguous_term_creation(self) -> None:
        """An ambiguous term should validate."""
        ambiguous = AmbiguousTerm(
            term="symbol",
            possible_canonicals=["Symbol (Coleridgean)", "Symbol (general)"],
            reason="Guite uses 'symbol' in both Coleridge-specific and general senses",
        )
        assert len(ambiguous.possible_canonicals) == 2

    def test_terminology_map_creation(self) -> None:
        """A complete terminology map should validate."""
        tm = TerminologyMap(
            author_id="test--guite",
            mappings=[
                TermMapping(
                    variant="Secondary Imagination",
                    canonical="Imagination",
                    relationship="narrower",
                    confidence=0.9,
                ),
            ],
            ambiguous=[],
            canonical_terms=["Imagination", "Sacrament"],
        )
        assert len(tm.mappings) == 1
        assert len(tm.canonical_terms) == 2

    def test_confidence_bounds(self) -> None:
        """Confidence must be between 0.0 and 1.0."""
        with pytest.raises(ValueError):
            TermMapping(
                variant="test",
                canonical="Test",
                confidence=1.5,
            )


# ---------------------------------------------------------------------------
# Term extraction tests
# ---------------------------------------------------------------------------


class TestTermExtraction:
    """Test term extraction from chunk texts."""

    def test_extracts_capitalized_phrases(self) -> None:
        """Should find capitalized multi-word phrases appearing at least twice."""
        chunks = [
            {"text": "Coleridge's Primary Imagination is important. Primary Imagination is key."},
            {"text": "Secondary Imagination echoes Primary Imagination."},
        ]
        terms = _extract_terms_from_chunks(chunks)
        assert "Primary Imagination" in terms

    def test_extracts_quoted_terms(self) -> None:
        """Should find quoted terms appearing multiple times."""
        chunks = [
            {"text": "He uses 'sacramental vision' throughout his work."},
            {"text": "The concept of 'sacramental vision' is central."},
        ]
        terms = _extract_terms_from_chunks(chunks)
        assert "sacramental vision" in terms

    def test_ignores_single_occurrence(self) -> None:
        """Terms appearing only once should be excluded."""
        chunks = [
            {"text": "The Unique Concept appears only here."},
        ]
        terms = _extract_terms_from_chunks(chunks)
        assert "Unique Concept" not in terms

    def test_extracts_from_real_literary_chunks(self) -> None:
        """Should extract meaningful terms from our sample literary text."""
        terms = _extract_terms_from_chunks(PRIMARY_CHUNKS)
        # These capitalized phrases appear in our sample data
        assert any("Imagination" in t for t in terms), (
            f"Expected 'Imagination' variant in terms, got: {terms[:10]}"
        )


# ---------------------------------------------------------------------------
# Lookup tests
# ---------------------------------------------------------------------------


class TestTermLookup:
    """Test the terminology lookup function."""

    def test_lookup_finds_mapping(self) -> None:
        """Should return canonical form for mapped variant."""
        tm = TerminologyMap(
            author_id="test",
            mappings=[
                TermMapping(
                    variant="imaginative faculty",
                    canonical="Imagination",
                    confidence=0.9,
                ),
            ],
            canonical_terms=["Imagination"],
        )
        normalizer = TerminologyNormalizer.__new__(TerminologyNormalizer)
        result = normalizer.lookup("imaginative faculty", tm)
        assert result == "Imagination"

    def test_lookup_case_insensitive(self) -> None:
        """Lookup should be case-insensitive."""
        tm = TerminologyMap(
            author_id="test",
            mappings=[
                TermMapping(
                    variant="sacramental",
                    canonical="Sacrament",
                    confidence=0.9,
                ),
            ],
            canonical_terms=["Sacrament"],
        )
        normalizer = TerminologyNormalizer.__new__(TerminologyNormalizer)
        result = normalizer.lookup("Sacramental", tm)
        assert result == "Sacrament"

    def test_lookup_returns_original_when_unmapped(self) -> None:
        """Should return original term when no mapping exists."""
        tm = TerminologyMap(
            author_id="test",
            mappings=[],
            canonical_terms=[],
        )
        normalizer = TerminologyNormalizer.__new__(TerminologyNormalizer)
        result = normalizer.lookup("unknown term", tm)
        assert result == "unknown term"


# ---------------------------------------------------------------------------
# Integration test (requires API key)
# ---------------------------------------------------------------------------


@requires_anthropic_key
async def test_terminology_normalization_integration(
    app_settings: Settings,
) -> None:
    """End-to-end terminology normalization against real API."""
    normalizer = TerminologyNormalizer(app_settings)

    result = await normalizer.normalize(
        author_id="test--guite",
        author_name="Malcolm Guite",
        chunks=PRIMARY_CHUNKS,
    )

    assert result.author_id == "test--guite"
    # Should have identified some canonical terms from the literary text
    assert len(result.canonical_terms) > 0 or len(result.mappings) > 0

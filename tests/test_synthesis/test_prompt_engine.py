"""Tests for O2: Synthesis prompt engineering."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from author_library.synthesis.gatherer import GatheredReflections, PersonalReflection
from author_library.synthesis.prompt_engine import (
    SourceCitation,
    SynthesisConfidence,
    SynthesisPromptEngine,
    SynthesisResult,
    _extract_citations,
    _split_sections,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_reflection(
    *,
    chunk_id: str = "ref-1",
    text: str = "I think imagination is a form of prayer.",
    date_created: str = "2026-01-15T10:30:00",
    metadata: dict | None = None,
) -> PersonalReflection:
    return PersonalReflection(
        chunk_id=chunk_id,
        work_id="personal--my-notes",
        text=text,
        date_created=date_created,
        granularity="micro",
        metadata=metadata or {},
    )


def _make_gathered(
    reflections: list[PersonalReflection] | None = None,
) -> GatheredReflections:
    refs = reflections or [
        _make_reflection(
            chunk_id="ref-1",
            text="I think Guite is right about imagination as prayer.",
            date_created="2026-01-15T10:30:00",
        ),
        _make_reflection(
            chunk_id="ref-2",
            text="But liturgical implications still trouble me.",
            date_created="2026-01-20T14:00:00",
        ),
        _make_reflection(
            chunk_id="ref-3",
            text="After reading Ordway, I see more clearly how imagination connects to truth.",
            date_created="2026-02-01T09:00:00",
        ),
    ]
    return GatheredReflections(
        reflections=refs,
        total_found=len(refs),
        filters_applied={"theme": "imagination"},
        date_range=("2026-01-15T10:30:00", "2026-02-01T09:00:00"),
    )


@pytest.fixture()
def mock_settings():
    settings = MagicMock()
    settings.llm.query_model = "claude-sonnet-4-5-20250929"
    settings.api_keys.anthropic_api_key.get_secret_value.return_value = "test-key"
    return settings


@pytest.fixture()
def engine(mock_settings):
    return SynthesisPromptEngine(settings=mock_settings)


# ---------------------------------------------------------------------------
# SynthesisConfidence tests
# ---------------------------------------------------------------------------


class TestSynthesisConfidence:
    def test_values(self):
        assert SynthesisConfidence.TENTATIVE == "tentative"
        assert SynthesisConfidence.DEVELOPING == "developing"
        assert SynthesisConfidence.COHERENT == "coherent"


# ---------------------------------------------------------------------------
# SynthesisResult tests
# ---------------------------------------------------------------------------


class TestSynthesisResult:
    def test_to_dict(self):
        result = SynthesisResult(
            synthesis="You appear to believe...",
            sources_used=[
                SourceCitation(
                    capture_id="ref-1",
                    note_path="personal--notes",
                    excerpt="I think...",
                    date="2026-01-15",
                ),
            ],
            confidence=SynthesisConfidence.DEVELOPING,
            open_tensions=["Liturgy vs imagination"],
            theme="imagination",
            prompt="What do I think?",
            reflection_count=3,
            date_range=("2026-01-15", "2026-02-01"),
        )
        d = result.to_dict()

        assert d["synthesis"] == "You appear to believe..."
        assert d["confidence"] == "developing"
        assert len(d["sources_used"]) == 1
        assert d["sources_used"][0]["capture_id"] == "ref-1"
        assert d["open_tensions"] == ["Liturgy vs imagination"]
        assert d["date_range"] == ["2026-01-15", "2026-02-01"]
        assert d["reflection_count"] == 3

    def test_to_dict_no_date_range(self):
        result = SynthesisResult(
            synthesis="",
            sources_used=[],
            confidence=SynthesisConfidence.TENTATIVE,
            open_tensions=[],
            theme="",
            prompt="",
            reflection_count=0,
        )
        assert result.to_dict()["date_range"] is None


# ---------------------------------------------------------------------------
# SourceCitation tests
# ---------------------------------------------------------------------------


class TestSourceCitation:
    def test_creation(self):
        cite = SourceCitation(
            capture_id="ref-1",
            note_path="personal--notes",
            excerpt="I think imagination is prayer.",
            date="2026-01-15",
        )
        assert cite.capture_id == "ref-1"
        assert cite.date == "2026-01-15"


# ---------------------------------------------------------------------------
# _split_sections tests
# ---------------------------------------------------------------------------


class TestSplitSections:
    def test_basic_split(self):
        raw = """## SYNTHESIS
Some synthesis text here.

## CONFIDENCE
developing

## OPEN_TENSIONS
- Tension one
- Tension two"""
        sections = _split_sections(raw)
        assert "SYNTHESIS" in sections
        assert "CONFIDENCE" in sections
        assert "OPEN_TENSIONS" in sections
        assert "synthesis text" in sections["SYNTHESIS"]

    def test_empty_input(self):
        assert _split_sections("") == {}

    def test_no_sections(self):
        sections = _split_sections("Just some plain text without headers.")
        assert sections == {}

    def test_section_with_multiple_lines(self):
        raw = """## SYNTHESIS
First paragraph.

Second paragraph.

## CONFIDENCE
coherent"""
        sections = _split_sections(raw)
        assert "First paragraph" in sections["SYNTHESIS"]
        assert "Second paragraph" in sections["SYNTHESIS"]


# ---------------------------------------------------------------------------
# _extract_citations tests
# ---------------------------------------------------------------------------


class TestExtractCitations:
    def test_basic_extraction(self):
        reflections = [
            _make_reflection(chunk_id="ref-1", text="First reflection text"),
            _make_reflection(chunk_id="ref-2", text="Second reflection text"),
        ]
        synthesis = "You noted [REF-1] and later elaborated [REF-2]."
        citations = _extract_citations(synthesis, reflections)

        assert len(citations) == 2
        assert citations[0].capture_id == "ref-1"
        assert citations[1].capture_id == "ref-2"

    def test_deduplicates_refs(self):
        reflections = [_make_reflection(chunk_id="ref-1")]
        synthesis = "[REF-1] is important. As mentioned in [REF-1]."
        citations = _extract_citations(synthesis, reflections)
        assert len(citations) == 1

    def test_ignores_invalid_refs(self):
        reflections = [_make_reflection(chunk_id="ref-1")]
        synthesis = "You said [REF-1] and also [REF-99]."
        citations = _extract_citations(synthesis, reflections)
        assert len(citations) == 1
        assert citations[0].capture_id == "ref-1"

    def test_no_refs(self):
        reflections = [_make_reflection()]
        citations = _extract_citations("No references here.", reflections)
        assert citations == []

    def test_excerpt_truncation(self):
        long_text = "A" * 500
        reflections = [_make_reflection(text=long_text)]
        synthesis = "[REF-1] was insightful."
        citations = _extract_citations(synthesis, reflections)
        assert len(citations[0].excerpt) <= 203  # 200 + "..."
        assert citations[0].excerpt.endswith("...")

    def test_date_extracted(self):
        reflections = [_make_reflection(date_created="2026-01-15T10:30:00")]
        synthesis = "[REF-1] noted."
        citations = _extract_citations(synthesis, reflections)
        assert citations[0].date == "2026-01-15"


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    def test_system_prompt_contains_rules(self, engine):
        prompt = engine._build_system_prompt("imagination", "What do I think?")
        assert "SYNTHESIS" in prompt
        assert "CONFIDENCE" in prompt
        assert "OPEN_TENSIONS" in prompt
        assert "second person" in prompt
        assert "quote" in prompt.lower()

    def test_user_prompt_contains_reflections(self, engine):
        gathered = _make_gathered()
        prompt = engine._build_user_prompt(
            gathered.reflections, "imagination", "What do I think?",
        )
        assert "[REF-1]" in prompt
        assert "[REF-2]" in prompt
        assert "[REF-3]" in prompt
        assert "imagination" in prompt.lower()
        assert "What do I think?" in prompt

    def test_user_prompt_truncates_long_text(self, engine):
        refs = [_make_reflection(text="X" * 2000)]
        prompt = engine._build_user_prompt(refs, "", "")
        assert "..." in prompt

    def test_user_prompt_no_prompt_or_theme(self, engine):
        refs = [_make_reflection()]
        prompt = engine._build_user_prompt(refs, "", "")
        assert "[REF-1]" in prompt


# ---------------------------------------------------------------------------
# Synthesize integration tests (with mocked LLM)
# ---------------------------------------------------------------------------


class TestSynthesize:
    @pytest.mark.asyncio()
    async def test_empty_reflections(self, engine):
        gathered = GatheredReflections(
            reflections=[], total_found=0, filters_applied={},
        )
        result = await engine.synthesize(gathered, theme="imagination")
        assert result.synthesis == ""
        assert result.confidence == SynthesisConfidence.TENTATIVE
        assert result.reflection_count == 0

    @pytest.mark.asyncio()
    async def test_full_synthesis(self, engine):
        gathered = _make_gathered()

        llm_response = """## SYNTHESIS
You appear to have moved from initial agreement with Guite's framework [REF-1] toward a more nuanced position. While you accept imagination-as-prayer theoretically, you maintain reservations about liturgical application [REF-2]. After engaging with Ordway's work [REF-3], you seem to be finding a way to reconcile these positions.

## SOURCES
REF-1: "I think Guite is right about imagination as prayer." (2026-01-15)
REF-2: "But liturgical implications still trouble me." (2026-01-20)
REF-3: "After reading Ordway, I see more clearly how imagination connects to truth." (2026-02-01)

## CONFIDENCE
developing

## OPEN_TENSIONS
- You accept imagination-as-prayer intellectually but resist its liturgical implications
- The role of truth in imaginative experience remains unresolved"""

        with patch.object(engine, "_call_llm", new_callable=AsyncMock, return_value=llm_response):
            result = await engine.synthesize(
                gathered, theme="imagination", prompt="What do I think about imagination?",
            )

        assert "Guite" in result.synthesis
        assert result.confidence == SynthesisConfidence.DEVELOPING
        assert len(result.open_tensions) == 2
        assert len(result.sources_used) == 3
        assert result.sources_used[0].capture_id == "ref-1"
        assert result.reflection_count == 3

    @pytest.mark.asyncio()
    async def test_coherent_confidence(self, engine):
        gathered = _make_gathered()
        llm_response = """## SYNTHESIS
Your position is clear.

## CONFIDENCE
coherent

## OPEN_TENSIONS
None identified."""

        with patch.object(engine, "_call_llm", new_callable=AsyncMock, return_value=llm_response):
            result = await engine.synthesize(gathered)

        assert result.confidence == SynthesisConfidence.COHERENT
        assert result.open_tensions == []

    @pytest.mark.asyncio()
    async def test_tentative_confidence(self, engine):
        gathered = _make_gathered([_make_reflection()])
        llm_response = """## SYNTHESIS
You have only one reflection so far. [REF-1]

## CONFIDENCE
tentative

## OPEN_TENSIONS
None identified."""

        with patch.object(engine, "_call_llm", new_callable=AsyncMock, return_value=llm_response):
            result = await engine.synthesize(gathered)

        assert result.confidence == SynthesisConfidence.TENTATIVE

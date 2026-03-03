"""Tests for O4: Open tension detector."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from author_library.synthesis.gatherer import GatheredReflections, PersonalReflection
from author_library.synthesis.tension_detector import (
    Tension,
    TensionAnalysis,
    TensionDetector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_reflection(
    *,
    chunk_id: str = "ref-1",
    text: str = "Some reflection",
    date_created: str = "2026-01-15",
) -> PersonalReflection:
    return PersonalReflection(
        chunk_id=chunk_id,
        work_id="personal--notes",
        text=text,
        date_created=date_created,
        granularity="micro",
    )


def _make_gathered(
    reflections: list[PersonalReflection] | None = None,
) -> GatheredReflections:
    refs = reflections or [
        _make_reflection(
            chunk_id="ref-1",
            text="I accept imagination as a form of prayer.",
            date_created="2026-01-15",
        ),
        _make_reflection(
            chunk_id="ref-2",
            text="I'm skeptical about the practical liturgical implications.",
            date_created="2026-01-20",
        ),
        _make_reflection(
            chunk_id="ref-3",
            text="After more reading, I'm coming around to the liturgical view.",
            date_created="2026-02-01",
        ),
    ]
    return GatheredReflections(
        reflections=refs,
        total_found=len(refs),
        filters_applied={"theme": "imagination"},
    )


@pytest.fixture()
def mock_settings():
    settings = MagicMock()
    settings.llm.query_model = "claude-sonnet-4-5-20250929"
    settings.api_keys.anthropic_api_key.get_secret_value.return_value = "test-key"
    return settings


@pytest.fixture()
def detector(mock_settings):
    return TensionDetector(settings=mock_settings)


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestTension:
    def test_creation(self):
        t = Tension(
            description="Conflicting views on liturgy",
            reflection_a_id="ref-1",
            reflection_b_id="ref-2",
            tension_type="contradiction",
            evidence_a="I accept imagination as prayer",
            evidence_b="I'm skeptical about liturgy",
            date_a="2026-01-15",
            date_b="2026-01-20",
        )
        assert t.tension_type == "contradiction"


class TestTensionAnalysis:
    def test_to_dict(self):
        analysis = TensionAnalysis(
            tensions=[
                Tension(
                    description="Liturgy tension",
                    reflection_a_id="ref-1",
                    reflection_b_id="ref-2",
                    tension_type="contradiction",
                    evidence_a="Accept",
                    evidence_b="Reject",
                    date_a="2026-01-15",
                    date_b="2026-01-20",
                ),
            ],
            reflection_count=3,
            theme="imagination",
        )
        d = analysis.to_dict()
        assert d["tension_count"] == 1
        assert d["reflection_count"] == 3
        assert d["theme"] == "imagination"

    def test_empty(self):
        analysis = TensionAnalysis(tensions=[], reflection_count=0, theme="")
        d = analysis.to_dict()
        assert d["tension_count"] == 0


# ---------------------------------------------------------------------------
# Prompt building tests
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    def test_system_prompt(self, detector):
        prompt = detector._build_system_prompt("imagination")
        assert "CONTRADICTIONS" in prompt
        assert "EVOLUTION" in prompt
        assert "TENSION" in prompt
        assert "REF" in prompt

    def test_user_prompt(self, detector):
        gathered = _make_gathered()
        prompt = detector._build_user_prompt(gathered.reflections, "imagination")
        assert "[REF-1]" in prompt
        assert "[REF-2]" in prompt
        assert "[REF-3]" in prompt
        assert "imagination" in prompt.lower()


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------


class TestResponseParsing:
    def test_parse_no_tensions(self, detector):
        gathered = _make_gathered()
        tensions = detector._parse_response("NO_TENSIONS_FOUND", gathered.reflections)
        assert tensions == []

    def test_parse_single_tension(self, detector):
        gathered = _make_gathered()
        raw = """TENSION: Conflicting views on liturgical implications
TYPE: contradiction
REF_A: 1 — I accept imagination as prayer
REF_B: 2 — I'm skeptical about liturgy"""

        tensions = detector._parse_response(raw, gathered.reflections)
        assert len(tensions) == 1
        assert tensions[0].tension_type == "contradiction"
        assert tensions[0].reflection_a_id == "ref-1"
        assert tensions[0].reflection_b_id == "ref-2"

    def test_parse_multiple_tensions(self, detector):
        gathered = _make_gathered()
        raw = """TENSION: Liturgy contradiction
TYPE: contradiction
REF_A: 1 — Accept prayer
REF_B: 2 — Skeptical of liturgy

TENSION: Position evolution on liturgy
TYPE: evolution
REF_A: 2 — Skeptical
REF_B: 3 — Coming around"""

        tensions = detector._parse_response(raw, gathered.reflections)
        assert len(tensions) == 2
        assert tensions[0].tension_type == "contradiction"
        assert tensions[1].tension_type == "evolution"

    def test_parse_invalid_ref(self, detector):
        """Invalid REF numbers are skipped."""
        gathered = _make_gathered()
        raw = """TENSION: Bad refs
TYPE: uncertainty
REF_A: 99 — Invalid
REF_B: 100 — Also invalid"""

        tensions = detector._parse_response(raw, gathered.reflections)
        assert len(tensions) == 0


# ---------------------------------------------------------------------------
# Full detect tests (with mocked LLM)
# ---------------------------------------------------------------------------


class TestDetect:
    @pytest.mark.asyncio()
    async def test_too_few_reflections(self, detector):
        """Less than 2 reflections — no tensions possible."""
        gathered = GatheredReflections(
            reflections=[_make_reflection()],
            total_found=1,
            filters_applied={},
        )
        result = await detector.detect(gathered, theme="imagination")
        assert result.tensions == []
        assert result.reflection_count == 1

    @pytest.mark.asyncio()
    async def test_full_detection(self, detector):
        gathered = _make_gathered()
        llm_response = """TENSION: Shifting view on liturgical implications
TYPE: evolution
REF_A: 2 — skeptical about liturgical implications
REF_B: 3 — coming around to the liturgical view"""

        from unittest.mock import patch

        with patch.object(detector, "_call_llm", new_callable=AsyncMock, return_value=llm_response):
            result = await detector.detect(gathered, theme="imagination")

        assert len(result.tensions) == 1
        assert result.tensions[0].tension_type == "evolution"
        assert result.tensions[0].reflection_a_id == "ref-2"
        assert result.tensions[0].reflection_b_id == "ref-3"
        assert result.theme == "imagination"

    @pytest.mark.asyncio()
    async def test_no_tensions_detected(self, detector):
        gathered = _make_gathered()

        from unittest.mock import patch

        with patch.object(detector, "_call_llm", new_callable=AsyncMock, return_value="NO_TENSIONS_FOUND"):
            result = await detector.detect(gathered, theme="imagination")

        assert result.tensions == []

"""Tests for context assembly and voice calibration."""

from __future__ import annotations

from uuid import uuid4

import pytest

from author_library.intelligence.thematic_index import ThematicAppearance, ThematicEntry
from author_library.intelligence.voice_profile import VoiceProfile
from author_library.retrieval.context_assembly import (
    _estimate_tokens,
    _format_voice_profile_text,
    assemble_context,
    build_voice_system_prompt,
)
from author_library.retrieval.models import (
    GraphExpansionResult,
    QuestionType,
    RetrievalResult,
)
from author_library.retrieval.orchestrator import OrchestratedResult

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def voice_profile() -> VoiceProfile:
    return VoiceProfile(
        author_id="cs-lewis",
        register="academic but accessible, with conversational warmth",
        sentence_patterns=[
            "favors long periodic sentences with embedded clauses",
            "alternates between short declarative and expansive constructions",
        ],
        vocabulary_tendencies=[
            "frequently uses 'sacramental', 'incarnational'",
            "prefers Anglo-Saxon roots over Latinate vocabulary",
        ],
        rhetorical_moves=[
            "builds arguments through close reading of literature",
            "uses analogy and parable to illustrate abstract concepts",
        ],
        characteristic_phrases=[
            "further up and further in",
            "the weight of glory",
            "surprised by joy",
        ],
        humor_style="dry, self-deprecating wit with occasional whimsy",
        example_passages=[
            "If we find ourselves with a desire that nothing in this world "
            "can satisfy, the most probable explanation is that we were made "
            "for another world."
        ],
        confidence=0.85,
    )


@pytest.fixture()
def thematic_entries() -> list[ThematicEntry]:
    return [
        ThematicEntry(
            theme="Joy / Sehnsucht",
            author_stance=(
                "Lewis treats joy (Sehnsucht) as a signpost pointing beyond "
                "the material world to the divine."
            ),
            appearances=[
                ThematicAppearance(
                    work_id="lewis--surprised-by-joy",
                    chapters=["Ch. 1", "Ch. 15"],
                    treatment_summary="Autobiographical account of his pursuit of joy",
                ),
            ],
            related_themes=["Desire", "Heaven", "Imagination"],
            key_passages=[],
        ),
    ]


@pytest.fixture()
def orchestrated_result() -> OrchestratedResult:
    """An OrchestratedResult with sample data across all three passes."""
    return OrchestratedResult(
        question="How does Lewis treat the concept of joy?",
        question_type=QuestionType.THEMATIC,
        primary_results=[
            RetrievalResult(
                chunk_id=uuid4(),
                work_id="lewis--surprised-by-joy",
                text=(
                    "The experience of Sehnsucht — inconsolable longing — "
                    "was the central thread of Lewis's spiritual autobiography."
                ),
                score=0.92,
                granularity="meso",
                source_class="primary",
                source="fusion",
            ),
            RetrievalResult(
                chunk_id=uuid4(),
                work_id="lewis--weight-of-glory",
                text=(
                    "The Weight of Glory argues that our longing for beauty "
                    "is a desire for the far-off country."
                ),
                score=0.88,
                granularity="meso",
                source_class="primary",
                source="fusion",
            ),
        ],
        graph_expansions=[
            GraphExpansionResult(
                chunk_id=str(uuid4()),
                work_id="macdonald--phantastes",
                text_preview=(
                    "MacDonald's fairy romance baptized Lewis's imagination "
                    "with the quality he later named joy."
                ),
                granularity="meso",
                source_class="contextual",
                relationship_type="ENGAGES_WITH",
                confidence="high",
                evidence="Lewis credits MacDonald as formative influence",
            ),
        ],
        supporting_evidence=[
            RetrievalResult(
                chunk_id=uuid4(),
                work_id="lewis--mere-christianity",
                text=(
                    "'If we find ourselves with a desire that nothing in this "
                    "world can satisfy...'"
                ),
                score=0.75,
                granularity="micro",
                source_class="primary",
                source="vector",
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Tests: Voice System Prompt
# ---------------------------------------------------------------------------


class TestVoiceSystemPrompt:
    """Tests for voice calibration system prompt generation."""

    def test_includes_author_name(self, voice_profile: VoiceProfile) -> None:
        prompt = build_voice_system_prompt(voice_profile, author_name="C.S. Lewis")
        assert "C.S. Lewis" in prompt

    def test_includes_register(self, voice_profile: VoiceProfile) -> None:
        prompt = build_voice_system_prompt(voice_profile, author_name="C.S. Lewis")
        assert "academic but accessible" in prompt

    def test_includes_patterns(self, voice_profile: VoiceProfile) -> None:
        prompt = build_voice_system_prompt(voice_profile, author_name="C.S. Lewis")
        assert "periodic sentences" in prompt

    def test_includes_example(self, voice_profile: VoiceProfile) -> None:
        prompt = build_voice_system_prompt(voice_profile, author_name="C.S. Lewis")
        assert "desire that nothing in this world" in prompt

    def test_source_class_labels(self, voice_profile: VoiceProfile) -> None:
        prompt = build_voice_system_prompt(voice_profile, author_name="C.S. Lewis")
        assert "PRIMARY" in prompt
        assert "SECONDARY" in prompt
        assert "CONTEXTUAL" in prompt

    def test_voice_contamination_warning(self, voice_profile: VoiceProfile) -> None:
        prompt = build_voice_system_prompt(voice_profile, author_name="C.S. Lewis")
        assert "Never present SECONDARY or CONTEXTUAL material" in prompt


# ---------------------------------------------------------------------------
# Tests: Voice Profile Formatting
# ---------------------------------------------------------------------------


class TestVoiceProfileText:
    """Tests for voice profile text formatting."""

    def test_includes_register(self, voice_profile: VoiceProfile) -> None:
        text = _format_voice_profile_text(voice_profile)
        assert "Register:" in text
        assert "academic but accessible" in text

    def test_includes_confidence(self, voice_profile: VoiceProfile) -> None:
        text = _format_voice_profile_text(voice_profile)
        assert "0.85" in text

    def test_includes_humor_style(self, voice_profile: VoiceProfile) -> None:
        text = _format_voice_profile_text(voice_profile)
        assert "Humor style:" in text
        assert "dry, self-deprecating" in text


# ---------------------------------------------------------------------------
# Tests: Context Assembly
# ---------------------------------------------------------------------------


class TestContextAssembly:
    """Tests for full context window assembly."""

    def test_assembles_with_voice_profile(
        self,
        orchestrated_result: OrchestratedResult,
        voice_profile: VoiceProfile,
        thematic_entries: list[ThematicEntry],
    ) -> None:
        """Full assembly includes voice profile, passages, and thematic summaries."""
        ctx = assemble_context(
            orchestrated_result,
            voice_profile=voice_profile,
            author_name="C.S. Lewis",
            thematic_entries=thematic_entries,
        )
        assert ctx.voice_profile_text
        assert ctx.system_prompt
        assert len(ctx.passages) > 0
        assert ctx.total_tokens_estimate > 0

    def test_assembles_without_voice_profile(
        self, orchestrated_result: OrchestratedResult
    ) -> None:
        """Assembly works without a voice profile (generic prompt)."""
        ctx = assemble_context(orchestrated_result, author_name="C.S. Lewis")
        assert "C.S. Lewis" in ctx.system_prompt
        assert ctx.voice_profile_text == ""

    def test_passages_include_source_class_labels(
        self, orchestrated_result: OrchestratedResult
    ) -> None:
        """Every passage text includes a source_class label."""
        ctx = assemble_context(orchestrated_result, author_name="C.S. Lewis")
        for passage in ctx.passages:
            # Label should be in the formatted text as [PRIMARY], [CONTEXTUAL], etc.
            assert any(
                label in passage.text
                for label in ["[PRIMARY]", "[SECONDARY]", "[CONTEXTUAL]", "[TERTIARY]"]
            )

    def test_token_budget_respected(
        self,
        orchestrated_result: OrchestratedResult,
        voice_profile: VoiceProfile,
    ) -> None:
        """Assembly respects token budget by trimming passages."""
        ctx = assemble_context(
            orchestrated_result,
            voice_profile=voice_profile,
            author_name="C.S. Lewis",
            token_budget=500,  # Very small budget
        )
        assert ctx.total_tokens_estimate <= 500

    def test_passages_ordered_by_priority(
        self, orchestrated_result: OrchestratedResult
    ) -> None:
        """Primary results come before graph expansions."""
        ctx = assemble_context(orchestrated_result, author_name="C.S. Lewis")
        sources = [p.source for p in ctx.passages]
        # Primary results (fusion) should precede graph expansion
        if "fusion" in sources and "graph" in sources:
            first_fusion = sources.index("fusion")
            first_graph = sources.index("graph")
            assert first_fusion < first_graph

    def test_thematic_summaries_included(
        self,
        orchestrated_result: OrchestratedResult,
        thematic_entries: list[ThematicEntry],
    ) -> None:
        """Thematic summaries are included when provided."""
        ctx = assemble_context(
            orchestrated_result,
            author_name="C.S. Lewis",
            thematic_entries=thematic_entries,
        )
        assert len(ctx.thematic_summaries) > 0
        assert "Joy" in ctx.thematic_summaries[0]

    def test_citation_labels(self, orchestrated_result: OrchestratedResult) -> None:
        """Each passage has a citation_label."""
        ctx = assemble_context(orchestrated_result, author_name="C.S. Lewis")
        for passage in ctx.passages:
            assert passage.citation_label
            assert "[" in passage.citation_label


# ---------------------------------------------------------------------------
# Tests: Token Estimation
# ---------------------------------------------------------------------------


class TestTokenEstimation:
    """Test the rough token estimator."""

    def test_empty_string(self) -> None:
        assert _estimate_tokens("") == 1  # Minimum 1

    def test_short_text(self) -> None:
        # "hello" = 5 chars / 4 = 1.25 -> 1
        assert _estimate_tokens("hello") == 1

    def test_longer_text(self) -> None:
        text = "This is a longer piece of text for testing token estimation."
        tokens = _estimate_tokens(text)
        assert tokens > 0
        assert tokens == len(text) // 4

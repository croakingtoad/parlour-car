"""Tests for O5: synthesize_my_thinking MCP tool handler."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from author_library.tools.synthesis import handle_synthesize_my_thinking


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_settings():
    settings = MagicMock()
    settings.llm.query_model = "claude-sonnet-4-5-20250929"
    settings.api_keys.anthropic_api_key.get_secret_value.return_value = "test-key"
    return settings


@pytest.fixture()
def mock_storage():
    storage = MagicMock()
    storage.pg = MagicMock()
    storage.neo4j = MagicMock()
    storage.works = MagicMock()
    storage.chunks = MagicMock()
    storage.embeddings = MagicMock()
    storage.graph = MagicMock()
    return storage


@pytest.fixture()
def mock_embedding_provider():
    return MagicMock()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHandleSynthesizeMyThinking:
    @pytest.mark.asyncio()
    async def test_missing_all_params_returns_error(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ):
        """At least one of theme/speaker/date_range/prompt required."""
        result = await handle_synthesize_my_thinking(
            {},
            settings=mock_settings,
            storage=mock_storage,
            embedding_provider=mock_embedding_provider,
        )
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio()
    async def test_no_reflections_returns_empty(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ):
        from author_library.synthesis.gatherer import GatheredReflections

        gathered = GatheredReflections(
            reflections=[],
            total_found=0,
            filters_applied={"theme": "imagination"},
        )

        with patch(
            "author_library.tools.synthesis.PersonalReflectionGatherer",
        ) as MockGatherer:
            instance = MockGatherer.return_value
            instance.gather = AsyncMock(return_value=gathered)

            result = await handle_synthesize_my_thinking(
                {"theme": "imagination"},
                settings=mock_settings,
                storage=mock_storage,
                embedding_provider=mock_embedding_provider,
            )

        parsed = json.loads(result)
        assert parsed["reflection_count"] == 0
        assert parsed["synthesis"] == ""
        assert "message" in parsed

    @pytest.mark.asyncio()
    async def test_full_pipeline_with_theme(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ):
        from author_library.synthesis.citation import CitationReport, EnrichedCitation
        from author_library.synthesis.gatherer import (
            GatheredReflections,
            PersonalReflection,
        )
        from author_library.synthesis.prompt_engine import (
            SynthesisConfidence,
            SynthesisResult,
        )
        from author_library.synthesis.tension_detector import (
            Tension,
            TensionAnalysis,
        )

        reflections = [
            PersonalReflection(
                chunk_id=f"ref-{i}",
                work_id="personal--notes",
                text=f"Reflection {i} about imagination",
                date_created=f"2026-01-{10 + i:02d}",
                granularity="micro",
                metadata={"themes": ["imagination"]},
            )
            for i in range(1, 4)
        ]

        gathered = GatheredReflections(
            reflections=reflections,
            total_found=3,
            filters_applied={"theme": "imagination"},
            date_range=("2026-01-11", "2026-01-13"),
        )

        synthesis = SynthesisResult(
            synthesis="You see imagination as prayer.",
            confidence=SynthesisConfidence.DEVELOPING,
            sources_used=[],
            open_tensions=["Early tension from O2"],
            theme="imagination",
            prompt="",
            reflection_count=3,
        )

        citation_report = CitationReport(
            citations=[
                EnrichedCitation(
                    capture_id="ref-1",
                    note_path="personal--notes/ref-1",
                    excerpt="Reflection 1 about imagination",
                    date="2026-01-11",
                    work_id="personal--notes",
                    work_title="Personal Notes",
                    author="",
                    section_type="my_response",
                    themes=["imagination"],
                    verified=True,
                ),
            ],
            total_citations=1,
            verified_count=1,
            unverified_count=0,
            unique_works=1,
        )

        tension_analysis = TensionAnalysis(
            tensions=[
                Tension(
                    description="Evolving view on imagination",
                    reflection_a_id="ref-1",
                    reflection_b_id="ref-3",
                    tension_type="evolution",
                    evidence_a="Early view",
                    evidence_b="Later view",
                    date_a="2026-01-11",
                    date_b="2026-01-13",
                ),
            ],
            reflection_count=3,
            theme="imagination",
        )

        with (
            patch("author_library.tools.synthesis.PersonalReflectionGatherer") as MockGatherer,
            patch("author_library.tools.synthesis.SynthesisPromptEngine") as MockEngine,
            patch("author_library.tools.synthesis.CitationEnricher") as MockEnricher,
            patch("author_library.tools.synthesis.TensionDetector") as MockDetector,
        ):
            MockGatherer.return_value.gather = AsyncMock(return_value=gathered)
            MockEngine.return_value.synthesize = AsyncMock(return_value=synthesis)
            MockEnricher.return_value.enrich = AsyncMock(return_value=citation_report)
            MockDetector.return_value.detect = AsyncMock(return_value=tension_analysis)

            result = await handle_synthesize_my_thinking(
                {"theme": "imagination"},
                settings=mock_settings,
                storage=mock_storage,
                embedding_provider=mock_embedding_provider,
            )

        parsed = json.loads(result)
        assert parsed["synthesis"] == "You see imagination as prayer."
        assert parsed["confidence"] == "developing"
        assert parsed["reflection_count"] == 3
        assert len(parsed["sources_used"]) == 1
        assert parsed["sources_used"][0]["capture_id"] == "ref-1"
        # O4 tensions preferred over O2 tensions
        assert len(parsed["open_tensions"]) == 1
        assert "Evolving view" in parsed["open_tensions"][0]
        assert parsed["theme_counts"] == {"imagination": 3}
        assert parsed["date_range"] == ["2026-01-11", "2026-01-13"]

    @pytest.mark.asyncio()
    async def test_o4_tensions_preferred_over_o2(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ):
        """When O4 detects tensions, they override O2 synthesis tensions."""
        from author_library.synthesis.citation import CitationReport
        from author_library.synthesis.gatherer import (
            GatheredReflections,
            PersonalReflection,
        )
        from author_library.synthesis.prompt_engine import (
            SynthesisConfidence,
            SynthesisResult,
        )
        from author_library.synthesis.tension_detector import (
            Tension,
            TensionAnalysis,
        )

        gathered = GatheredReflections(
            reflections=[
                PersonalReflection(
                    chunk_id="ref-1", work_id="personal--notes",
                    text="Reflection", date_created="2026-01-15",
                    granularity="micro",
                ),
                PersonalReflection(
                    chunk_id="ref-2", work_id="personal--notes",
                    text="Another", date_created="2026-01-20",
                    granularity="micro",
                ),
            ],
            total_found=2,
            filters_applied={"theme": "test"},
        )

        synthesis = SynthesisResult(
            synthesis="Position statement.",
            confidence=SynthesisConfidence.TENTATIVE,
            sources_used=[],
            open_tensions=["O2 tension A", "O2 tension B"],
            theme="test",
            prompt="",
            reflection_count=2,
        )

        citation_report = CitationReport(
            citations=[], total_citations=0,
            verified_count=0, unverified_count=0, unique_works=0,
        )

        # O4 finds one specific tension
        tension_analysis = TensionAnalysis(
            tensions=[
                Tension(
                    description="O4 specific tension",
                    reflection_a_id="ref-1",
                    reflection_b_id="ref-2",
                    tension_type="contradiction",
                    evidence_a="A", evidence_b="B",
                    date_a="2026-01-15", date_b="2026-01-20",
                ),
            ],
            reflection_count=2,
            theme="test",
        )

        with (
            patch("author_library.tools.synthesis.PersonalReflectionGatherer") as MockGatherer,
            patch("author_library.tools.synthesis.SynthesisPromptEngine") as MockEngine,
            patch("author_library.tools.synthesis.CitationEnricher") as MockEnricher,
            patch("author_library.tools.synthesis.TensionDetector") as MockDetector,
        ):
            MockGatherer.return_value.gather = AsyncMock(return_value=gathered)
            MockEngine.return_value.synthesize = AsyncMock(return_value=synthesis)
            MockEnricher.return_value.enrich = AsyncMock(return_value=citation_report)
            MockDetector.return_value.detect = AsyncMock(return_value=tension_analysis)

            result = await handle_synthesize_my_thinking(
                {"theme": "test"},
                settings=mock_settings,
                storage=mock_storage,
                embedding_provider=mock_embedding_provider,
            )

        parsed = json.loads(result)
        # O4 tensions replace O2 tensions
        assert len(parsed["open_tensions"]) == 1
        assert parsed["open_tensions"][0] == "O4 specific tension"

    @pytest.mark.asyncio()
    async def test_o2_tensions_fallback_when_o4_empty(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ):
        """When O4 finds no tensions, O2 synthesis tensions are used."""
        from author_library.synthesis.citation import CitationReport
        from author_library.synthesis.gatherer import (
            GatheredReflections,
            PersonalReflection,
        )
        from author_library.synthesis.prompt_engine import (
            SynthesisConfidence,
            SynthesisResult,
        )
        from author_library.synthesis.tension_detector import TensionAnalysis

        gathered = GatheredReflections(
            reflections=[
                PersonalReflection(
                    chunk_id="ref-1", work_id="personal--notes",
                    text="Reflection", date_created="2026-01-15",
                    granularity="micro",
                ),
                PersonalReflection(
                    chunk_id="ref-2", work_id="personal--notes",
                    text="Another", date_created="2026-01-20",
                    granularity="micro",
                ),
            ],
            total_found=2,
            filters_applied={"theme": "test"},
        )

        synthesis = SynthesisResult(
            synthesis="Position.",
            confidence=SynthesisConfidence.TENTATIVE,
            sources_used=[],
            open_tensions=["Fallback tension from O2"],
            theme="test",
            prompt="",
            reflection_count=2,
        )

        citation_report = CitationReport(
            citations=[], total_citations=0,
            verified_count=0, unverified_count=0, unique_works=0,
        )

        tension_analysis = TensionAnalysis(
            tensions=[], reflection_count=2, theme="test",
        )

        with (
            patch("author_library.tools.synthesis.PersonalReflectionGatherer") as MockGatherer,
            patch("author_library.tools.synthesis.SynthesisPromptEngine") as MockEngine,
            patch("author_library.tools.synthesis.CitationEnricher") as MockEnricher,
            patch("author_library.tools.synthesis.TensionDetector") as MockDetector,
        ):
            MockGatherer.return_value.gather = AsyncMock(return_value=gathered)
            MockEngine.return_value.synthesize = AsyncMock(return_value=synthesis)
            MockEnricher.return_value.enrich = AsyncMock(return_value=citation_report)
            MockDetector.return_value.detect = AsyncMock(return_value=tension_analysis)

            result = await handle_synthesize_my_thinking(
                {"theme": "test"},
                settings=mock_settings,
                storage=mock_storage,
                embedding_provider=mock_embedding_provider,
            )

        parsed = json.loads(result)
        assert parsed["open_tensions"] == ["Fallback tension from O2"]

    @pytest.mark.asyncio()
    async def test_with_prompt_and_date_range(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ):
        from author_library.synthesis.gatherer import GatheredReflections

        gathered = GatheredReflections(
            reflections=[],
            total_found=0,
            filters_applied={"prompt": "test"},
        )

        with patch(
            "author_library.tools.synthesis.PersonalReflectionGatherer",
        ) as MockGatherer:
            instance = MockGatherer.return_value
            instance.gather = AsyncMock(return_value=gathered)

            result = await handle_synthesize_my_thinking(
                {
                    "prompt": "What do I think about prayer?",
                    "date_range": {"after": "2026-01-01", "before": "2026-02-01"},
                },
                settings=mock_settings,
                storage=mock_storage,
                embedding_provider=mock_embedding_provider,
            )

            # Verify gather was called with correct params
            call_kwargs = instance.gather.call_args[1]
            assert call_kwargs["prompt"] == "What do I think about prayer?"
            assert call_kwargs["date_after"] == "2026-01-01"
            assert call_kwargs["date_before"] == "2026-02-01"

    @pytest.mark.asyncio()
    async def test_with_speaker(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ):
        from author_library.synthesis.gatherer import GatheredReflections

        gathered = GatheredReflections(
            reflections=[],
            total_found=0,
            filters_applied={"speaker": "guite"},
        )

        with patch(
            "author_library.tools.synthesis.PersonalReflectionGatherer",
        ) as MockGatherer:
            instance = MockGatherer.return_value
            instance.gather = AsyncMock(return_value=gathered)

            await handle_synthesize_my_thinking(
                {"speaker": "guite"},
                settings=mock_settings,
                storage=mock_storage,
                embedding_provider=mock_embedding_provider,
            )

            call_kwargs = instance.gather.call_args[1]
            assert call_kwargs["speaker"] == "guite"

    @pytest.mark.asyncio()
    async def test_citation_verification_in_response(
        self, mock_settings, mock_storage, mock_embedding_provider,
    ):
        from author_library.synthesis.citation import CitationReport, EnrichedCitation
        from author_library.synthesis.gatherer import (
            GatheredReflections,
            PersonalReflection,
        )
        from author_library.synthesis.prompt_engine import (
            SynthesisConfidence,
            SynthesisResult,
        )
        from author_library.synthesis.tension_detector import TensionAnalysis

        gathered = GatheredReflections(
            reflections=[
                PersonalReflection(
                    chunk_id="ref-1", work_id="personal--notes",
                    text="A thought", date_created="2026-01-15",
                    granularity="micro",
                ),
                PersonalReflection(
                    chunk_id="ref-2", work_id="personal--notes",
                    text="Another thought", date_created="2026-01-20",
                    granularity="micro",
                ),
                PersonalReflection(
                    chunk_id="ref-3", work_id="personal--notes",
                    text="Third thought", date_created="2026-01-25",
                    granularity="micro",
                ),
            ],
            total_found=3,
            filters_applied={"theme": "prayer"},
        )

        synthesis = SynthesisResult(
            synthesis="Your view on prayer.",
            confidence=SynthesisConfidence.COHERENT,
            sources_used=[],
            open_tensions=[],
            theme="prayer",
            prompt="",
            reflection_count=3,
        )

        citation_report = CitationReport(
            citations=[
                EnrichedCitation(
                    capture_id="ref-1", note_path="p/ref-1",
                    excerpt="A thought", date="2026-01-15",
                    work_id="personal--notes", work_title="Personal Notes",
                    author="", section_type="my_response",
                    themes=["prayer"], verified=True,
                ),
                EnrichedCitation(
                    capture_id="ref-2", note_path="p/ref-2",
                    excerpt="Another thought", date="2026-01-20",
                    work_id="personal--notes", work_title="Personal Notes",
                    author="", section_type="my_response",
                    themes=["prayer"], verified=False,
                ),
            ],
            total_citations=2,
            verified_count=1,
            unverified_count=1,
            unique_works=1,
        )

        tension_analysis = TensionAnalysis(
            tensions=[], reflection_count=3, theme="prayer",
        )

        with (
            patch("author_library.tools.synthesis.PersonalReflectionGatherer") as MockGatherer,
            patch("author_library.tools.synthesis.SynthesisPromptEngine") as MockEngine,
            patch("author_library.tools.synthesis.CitationEnricher") as MockEnricher,
            patch("author_library.tools.synthesis.TensionDetector") as MockDetector,
        ):
            MockGatherer.return_value.gather = AsyncMock(return_value=gathered)
            MockEngine.return_value.synthesize = AsyncMock(return_value=synthesis)
            MockEnricher.return_value.enrich = AsyncMock(return_value=citation_report)
            MockDetector.return_value.detect = AsyncMock(return_value=tension_analysis)

            result = await handle_synthesize_my_thinking(
                {"theme": "prayer"},
                settings=mock_settings,
                storage=mock_storage,
                embedding_provider=mock_embedding_provider,
            )

        parsed = json.loads(result)
        assert parsed["citation_verification"]["verified"] == 1
        assert parsed["citation_verification"]["unverified"] == 1
        assert parsed["citation_verification"]["unique_works"] == 1
        assert parsed["confidence"] == "coherent"

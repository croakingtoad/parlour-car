"""Tests for O6: Socratic loop — user response → Personal data → re-synthesis."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from author_library.synthesis.socratic import SocraticLoop, SocraticResponse, _to_json


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
    storage.graph = MagicMock()
    storage.embeddings = MagicMock()
    return storage


@pytest.fixture()
def mock_embedding_provider():
    provider = MagicMock()
    provider.embed_text = AsyncMock(return_value=[0.1] * 1024)
    return provider


@pytest.fixture()
def socratic_loop(mock_settings, mock_storage, mock_embedding_provider):
    return SocraticLoop(
        settings=mock_settings,
        storage=mock_storage,
        embedding_provider=mock_embedding_provider,
    )


# ---------------------------------------------------------------------------
# Data model tests
# ---------------------------------------------------------------------------


class TestSocraticResponse:
    def test_to_dict_minimal(self):
        resp = SocraticResponse(
            response_chunk_id="chunk-1",
            original_synthesis_theme="imagination",
            response_stored=True,
            message="Stored.",
        )
        d = resp.to_dict()
        assert d["response_chunk_id"] == "chunk-1"
        assert d["original_synthesis_theme"] == "imagination"
        assert d["response_stored"] is True
        assert d["message"] == "Stored."
        assert "re_synthesis" not in d

    def test_to_dict_with_re_synthesis(self):
        from author_library.synthesis.prompt_engine import (
            SynthesisConfidence,
            SynthesisResult,
        )

        synth = SynthesisResult(
            synthesis="Updated position.",
            confidence=SynthesisConfidence.DEVELOPING,
            sources_used=[],
            open_tensions=[],
            theme="liturgy",
            prompt="",
            reflection_count=1,
        )
        resp = SocraticResponse(
            response_chunk_id="chunk-2",
            original_synthesis_theme="liturgy",
            response_stored=True,
            re_synthesis=synth,
            message="Stored and re-synthesized.",
        )
        d = resp.to_dict()
        assert "re_synthesis" in d
        assert d["re_synthesis"]["synthesis"] == "Updated position."

    def test_to_dict_not_stored(self):
        resp = SocraticResponse(
            response_chunk_id="",
            original_synthesis_theme="test",
            response_stored=False,
            message="Empty response.",
        )
        d = resp.to_dict()
        assert d["response_stored"] is False
        assert d["response_chunk_id"] == ""


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestToJson:
    def test_dict_to_json(self):
        import json

        data = {"section_type": "my_response", "themes": ["imagination"]}
        result = _to_json(data)
        parsed = json.loads(result)
        assert parsed["section_type"] == "my_response"
        assert parsed["themes"] == ["imagination"]

    def test_empty_dict(self):
        import json

        result = _to_json({})
        assert json.loads(result) == {}


# ---------------------------------------------------------------------------
# process_response tests
# ---------------------------------------------------------------------------


class TestProcessResponse:
    @pytest.mark.asyncio()
    async def test_empty_response_not_stored(self, socratic_loop):
        result = await socratic_loop.process_response(
            user_response="",
            theme="imagination",
        )
        assert result.response_stored is False
        assert result.response_chunk_id == ""
        assert "Empty" in result.message

    @pytest.mark.asyncio()
    async def test_whitespace_only_not_stored(self, socratic_loop):
        result = await socratic_loop.process_response(
            user_response="   \n  ",
            theme="imagination",
        )
        assert result.response_stored is False

    @pytest.mark.asyncio()
    async def test_successful_storage(self, socratic_loop, mock_storage):
        mock_storage.pg.fetch_val = AsyncMock(return_value="new-chunk-id")
        mock_storage.graph.create_user_reflects_on_edge = AsyncMock()
        mock_storage.embeddings.store = AsyncMock()

        result = await socratic_loop.process_response(
            user_response="I think imagination is central to prayer.",
            theme="imagination",
            re_synthesize=False,
        )

        assert result.response_stored is True
        assert result.response_chunk_id == "new-chunk-id"
        assert result.original_synthesis_theme == "imagination"
        assert result.re_synthesis is None

    @pytest.mark.asyncio()
    async def test_storage_creates_personal_chunk(self, socratic_loop, mock_storage):
        mock_storage.pg.fetch_val = AsyncMock(return_value="chunk-99")
        mock_storage.graph.create_user_reflects_on_edge = AsyncMock()
        mock_storage.embeddings.store = AsyncMock()

        await socratic_loop.process_response(
            user_response="My reflection on liturgy.",
            theme="liturgy",
            re_synthesize=False,
        )

        # Verify the INSERT call
        call_args = mock_storage.pg.fetch_val.call_args
        sql = call_args[0][0]
        assert "INSERT INTO chunks" in sql
        assert "'personal'" in sql
        assert "'micro'" in sql

    @pytest.mark.asyncio()
    async def test_graph_edge_created_with_synthesis_ref(
        self, socratic_loop, mock_storage,
    ):
        mock_storage.pg.fetch_val = AsyncMock(return_value="chunk-1")
        mock_storage.graph.create_user_reflects_on_edge = AsyncMock()
        mock_storage.embeddings.store = AsyncMock()

        await socratic_loop.process_response(
            user_response="My response.",
            theme="imagination",
            synthesis_chunk_id="synth-chunk-42",
            re_synthesize=False,
        )

        mock_storage.graph.create_user_reflects_on_edge.assert_called_once_with(
            personal_chunk_id="chunk-1",
            target_id="synth-chunk-42",
            target_key="chunk_id",
            target_label="Chunk",
        )

    @pytest.mark.asyncio()
    async def test_graph_edge_not_created_without_synthesis_ref(
        self, socratic_loop, mock_storage,
    ):
        mock_storage.pg.fetch_val = AsyncMock(return_value="chunk-1")
        mock_storage.graph.create_user_reflects_on_edge = AsyncMock()
        mock_storage.embeddings.store = AsyncMock()

        await socratic_loop.process_response(
            user_response="My response.",
            theme="imagination",
            re_synthesize=False,
        )

        mock_storage.graph.create_user_reflects_on_edge.assert_not_called()

    @pytest.mark.asyncio()
    async def test_graph_edge_failure_graceful(
        self, socratic_loop, mock_storage,
    ):
        mock_storage.pg.fetch_val = AsyncMock(return_value="chunk-1")
        mock_storage.graph.create_user_reflects_on_edge = AsyncMock(
            side_effect=RuntimeError("Graph down"),
        )
        mock_storage.embeddings.store = AsyncMock()

        result = await socratic_loop.process_response(
            user_response="My response.",
            theme="imagination",
            synthesis_chunk_id="synth-1",
            re_synthesize=False,
        )

        # Storage succeeds even if graph edge fails
        assert result.response_stored is True
        assert result.response_chunk_id == "chunk-1"

    @pytest.mark.asyncio()
    async def test_embedding_failure_graceful(
        self, socratic_loop, mock_storage, mock_embedding_provider,
    ):
        mock_storage.pg.fetch_val = AsyncMock(return_value="chunk-1")
        mock_storage.graph.create_user_reflects_on_edge = AsyncMock()
        mock_embedding_provider.embed_text = AsyncMock(
            side_effect=RuntimeError("Embedding service down"),
        )

        result = await socratic_loop.process_response(
            user_response="My response.",
            theme="test",
            re_synthesize=False,
        )

        # Storage succeeds even if embedding fails
        assert result.response_stored is True

    @pytest.mark.asyncio()
    async def test_storage_failure_returns_not_stored(
        self, socratic_loop, mock_storage,
    ):
        mock_storage.pg.fetch_val = AsyncMock(
            side_effect=RuntimeError("DB down"),
        )

        result = await socratic_loop.process_response(
            user_response="My response.",
            theme="test",
            re_synthesize=False,
        )

        assert result.response_stored is False
        assert result.response_chunk_id == ""
        assert "Failed" in result.message

    @pytest.mark.asyncio()
    async def test_storage_returns_none_chunk_id(
        self, socratic_loop, mock_storage,
    ):
        mock_storage.pg.fetch_val = AsyncMock(return_value=None)

        result = await socratic_loop.process_response(
            user_response="My response.",
            theme="test",
            re_synthesize=False,
        )

        assert result.response_stored is False

    @pytest.mark.asyncio()
    async def test_re_synthesis_triggered(
        self, socratic_loop, mock_storage,
    ):
        mock_storage.pg.fetch_val = AsyncMock(return_value="chunk-1")
        mock_storage.graph.create_user_reflects_on_edge = AsyncMock()
        mock_storage.embeddings.store = AsyncMock()

        from author_library.synthesis.prompt_engine import (
            SynthesisConfidence,
            SynthesisResult,
        )

        synth = SynthesisResult(
            synthesis="Updated position incorporating your new reflection.",
            confidence=SynthesisConfidence.DEVELOPING,
            sources_used=[],
            open_tensions=[],
            theme="imagination",
            prompt="What do I think about imagination?",
            reflection_count=1,
        )

        with (
            patch("author_library.synthesis.socratic.PersonalReflectionGatherer") as MockGatherer,
            patch("author_library.synthesis.socratic.SynthesisPromptEngine") as MockEngine,
        ):
            from author_library.synthesis.gatherer import GatheredReflections, PersonalReflection

            gathered = GatheredReflections(
                reflections=[
                    PersonalReflection(
                        chunk_id="ref-1", work_id="personal--notes",
                        text="Earlier thought.", date_created="2026-01-15",
                        granularity="micro",
                    ),
                ],
                total_found=1,
                filters_applied={"theme": "imagination"},
            )
            MockGatherer.return_value.gather = AsyncMock(return_value=gathered)
            MockEngine.return_value.synthesize = AsyncMock(return_value=synth)

            result = await socratic_loop.process_response(
                user_response="Imagination IS prayer.",
                theme="imagination",
                re_synthesize=True,
            )

        assert result.response_stored is True
        assert result.re_synthesis is not None
        assert result.re_synthesis.synthesis == "Updated position incorporating your new reflection."

    @pytest.mark.asyncio()
    async def test_re_synthesis_failure_graceful(
        self, socratic_loop, mock_storage,
    ):
        mock_storage.pg.fetch_val = AsyncMock(return_value="chunk-1")
        mock_storage.graph.create_user_reflects_on_edge = AsyncMock()
        mock_storage.embeddings.store = AsyncMock()

        with patch(
            "author_library.synthesis.socratic.PersonalReflectionGatherer",
        ) as MockGatherer:
            MockGatherer.return_value.gather = AsyncMock(
                side_effect=RuntimeError("LLM down"),
            )

            result = await socratic_loop.process_response(
                user_response="My thoughts.",
                theme="imagination",
                re_synthesize=True,
            )

        # Storage succeeds, re-synthesis fails gracefully
        assert result.response_stored is True
        assert result.re_synthesis is None

    @pytest.mark.asyncio()
    async def test_custom_prompt_passed_to_re_synthesis(
        self, socratic_loop, mock_storage,
    ):
        mock_storage.pg.fetch_val = AsyncMock(return_value="chunk-1")
        mock_storage.graph.create_user_reflects_on_edge = AsyncMock()
        mock_storage.embeddings.store = AsyncMock()

        from author_library.synthesis.gatherer import GatheredReflections

        gathered = GatheredReflections(
            reflections=[], total_found=0, filters_applied={},
        )

        with (
            patch("author_library.synthesis.socratic.PersonalReflectionGatherer") as MockGatherer,
            patch("author_library.synthesis.socratic.SynthesisPromptEngine"),
        ):
            MockGatherer.return_value.gather = AsyncMock(return_value=gathered)

            await socratic_loop.process_response(
                user_response="My view.",
                theme="prayer",
                prompt="How has my view on prayer changed?",
                re_synthesize=True,
            )

            call_kwargs = MockGatherer.return_value.gather.call_args[1]
            assert call_kwargs["prompt"] == "How has my view on prayer changed?"

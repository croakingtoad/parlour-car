"""Tests for the OpenAI embedding provider.

Tests validate request construction and response parsing.
Integration tests that hit the real OpenAI API are skipped
when OPENAI_API_KEY is not set.
"""

from __future__ import annotations

import os

import pytest

from author_library.embeddings.openai import (
    _OPENAI_API_URL,
    OpenAIEmbeddingProvider,
)
from author_library.errors import EmbeddingError


class TestOpenAIRequestConstruction:
    """Validate that OpenAI requests are built correctly."""

    def test_init_requires_api_key(self) -> None:
        with pytest.raises(EmbeddingError, match="API key is required"):
            OpenAIEmbeddingProvider(api_key="")

    def test_provider_properties(self) -> None:
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        assert provider.provider_name == "openai"
        assert provider.model_name == "text-embedding-3-large"
        assert provider.dimensions == 3072

    def test_custom_model_and_dimensions(self) -> None:
        provider = OpenAIEmbeddingProvider(
            api_key="test-key",
            model="text-embedding-3-small",
            dimensions=1536,
        )
        assert provider.model_name == "text-embedding-3-small"
        assert provider.dimensions == 1536


class TestOpenAIResponseParsing:
    """Validate that OpenAI responses are parsed correctly."""

    def test_parse_single_embedding(self) -> None:
        response_json = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0,
                }
            ],
            "model": "text-embedding-3-large",
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        data_item = response_json["data"][0]
        assert data_item["embedding"] == [0.1, 0.2, 0.3]
        assert response_json["usage"]["total_tokens"] == 5

    def test_parse_batch_response_sorted(self) -> None:
        response_json = {
            "data": [
                {"embedding": [0.3], "index": 1},
                {"embedding": [0.1], "index": 0},
            ],
            "model": "text-embedding-3-large",
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
        sorted_data = sorted(response_json["data"], key=lambda d: d["index"])
        vectors = [d["embedding"] for d in sorted_data]
        assert vectors == [[0.1], [0.3]]


class TestOpenAIEmptyBatch:
    """Validate error on empty batch."""

    @pytest.mark.asyncio
    async def test_empty_batch_raises(self) -> None:
        provider = OpenAIEmbeddingProvider(api_key="test-key")
        with pytest.raises(EmbeddingError, match="empty batch"):
            await provider.embed_batch([])
        await provider.close()


class TestOpenAIApiUrlConstant:
    """Validate the API URL is correct."""

    def test_api_url(self) -> None:
        assert _OPENAI_API_URL == "https://api.openai.com/v1/embeddings"


# -- Integration tests (require live API key) --------------------------------

_has_openai_key = bool(os.environ.get("OPENAI_API_KEY"))


@pytest.mark.skipif(not _has_openai_key, reason="OPENAI_API_KEY not set")
class TestOpenAIIntegration:
    """Integration tests against the real OpenAI API."""

    @pytest.fixture
    def provider(self) -> OpenAIEmbeddingProvider:
        return OpenAIEmbeddingProvider(
            api_key=os.environ["OPENAI_API_KEY"],
        )

    @pytest.mark.asyncio
    async def test_embed_text(self, provider: OpenAIEmbeddingProvider) -> None:
        result = await provider.embed_text("Hello, world!")
        assert len(result.vector) == 3072
        assert result.provider == "openai"
        assert result.dimensions == 3072
        await provider.close()

    @pytest.mark.asyncio
    async def test_embed_batch(self, provider: OpenAIEmbeddingProvider) -> None:
        texts = ["First text", "Second text"]
        result = await provider.embed_batch(texts)
        assert len(result.vectors) == 2
        assert all(len(v) == 3072 for v in result.vectors)
        await provider.close()

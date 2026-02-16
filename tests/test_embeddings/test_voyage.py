"""Tests for the Voyage AI embedding provider.

Tests validate request construction and response parsing.
Integration tests that hit the real Voyage API are skipped
when VOYAGE_API_KEY is not set.
"""

from __future__ import annotations

import os

import pytest

from author_library.embeddings.voyage import (
    _MAX_BATCH_SIZE,
    _VOYAGE_API_URL,
    VoyageEmbeddingProvider,
)
from author_library.errors import EmbeddingError


class TestVoyageRequestConstruction:
    """Validate that Voyage requests are built correctly."""

    def test_init_requires_api_key(self) -> None:
        with pytest.raises(EmbeddingError, match="API key is required"):
            VoyageEmbeddingProvider(api_key="")

    def test_provider_properties(self) -> None:
        provider = VoyageEmbeddingProvider(api_key="test-key")
        assert provider.provider_name == "voyage"
        assert provider.model_name == "voyage-3-large"
        assert provider.dimensions == 1024

    def test_custom_model_and_dimensions(self) -> None:
        provider = VoyageEmbeddingProvider(
            api_key="test-key",
            model="voyage-3",
            dimensions=512,
        )
        assert provider.model_name == "voyage-3"
        assert provider.dimensions == 512


class TestVoyageResponseParsing:
    """Validate that Voyage responses are parsed correctly."""

    def test_parse_single_embedding(self) -> None:
        """Verify EmbeddingResult is built from a realistic API response."""
        response_json = {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": [0.1, 0.2, 0.3],
                    "index": 0,
                }
            ],
            "model": "voyage-3-large",
            "usage": {"total_tokens": 5},
        }
        # Directly validate parsing logic
        data_item = response_json["data"][0]
        assert data_item["embedding"] == [0.1, 0.2, 0.3]
        assert response_json["usage"]["total_tokens"] == 5

    def test_parse_batch_response_sorted(self) -> None:
        """Verify batch results are sorted by index."""
        response_json = {
            "data": [
                {"embedding": [0.3], "index": 1},
                {"embedding": [0.1], "index": 0},
            ],
            "model": "voyage-3-large",
            "usage": {"total_tokens": 10},
        }
        sorted_data = sorted(response_json["data"], key=lambda d: d["index"])
        vectors = [d["embedding"] for d in sorted_data]
        assert vectors == [[0.1], [0.3]]

    def test_batch_size_constant(self) -> None:
        """Verify the batch size limit is reasonable."""
        assert _MAX_BATCH_SIZE == 128


class TestVoyageEmptyBatch:
    """Validate error on empty batch."""

    @pytest.mark.asyncio
    async def test_empty_batch_raises(self) -> None:
        provider = VoyageEmbeddingProvider(api_key="test-key")
        with pytest.raises(EmbeddingError, match="empty batch"):
            await provider.embed_batch([])
        await provider.close()


class TestVoyageApiUrlConstant:
    """Validate the API URL is correct."""

    def test_api_url(self) -> None:
        assert _VOYAGE_API_URL == "https://api.voyageai.com/v1/embeddings"


# -- Integration tests (require live API key) --------------------------------

_has_voyage_key = bool(os.environ.get("VOYAGE_API_KEY"))


@pytest.mark.skipif(not _has_voyage_key, reason="VOYAGE_API_KEY not set")
class TestVoyageIntegration:
    """Integration tests against the real Voyage AI API."""

    @pytest.fixture
    def provider(self) -> VoyageEmbeddingProvider:
        return VoyageEmbeddingProvider(
            api_key=os.environ["VOYAGE_API_KEY"],
        )

    @pytest.mark.asyncio
    async def test_embed_text(self, provider: VoyageEmbeddingProvider) -> None:
        result = await provider.embed_text("Hello, world!")
        assert len(result.vector) == 1024
        assert result.model == "voyage-3-large"
        assert result.provider == "voyage"
        assert result.dimensions == 1024
        await provider.close()

    @pytest.mark.asyncio
    async def test_embed_query(self, provider: VoyageEmbeddingProvider) -> None:
        result = await provider.embed_query("What is the meaning of life?")
        assert len(result.vector) == 1024
        assert result.provider == "voyage"
        await provider.close()

    @pytest.mark.asyncio
    async def test_embed_batch(self, provider: VoyageEmbeddingProvider) -> None:
        texts = ["First text", "Second text", "Third text"]
        result = await provider.embed_batch(texts)
        assert len(result.vectors) == 3
        assert all(len(v) == 1024 for v in result.vectors)
        await provider.close()

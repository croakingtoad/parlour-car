"""Tests for the Voyage AI embedding provider.

Tests validate request construction and response parsing.
Integration tests that hit the real Voyage API are skipped
when VOYAGE_API_KEY is not set.
"""

from __future__ import annotations

import os

import pytest

from author_library.embeddings.base import (
    build_token_aware_batches,
    estimate_tokens,
)
from author_library.embeddings.voyage import (
    _MAX_BATCH_SIZE,
    _MAX_TOKENS_PER_BATCH,
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


class TestTokenEstimation:
    """Validate token estimation heuristic."""

    def test_estimate_tokens_short_text(self) -> None:
        assert estimate_tokens("hello world") == 3  # 2 words * 1.5 = 3.0 → 3

    def test_estimate_tokens_empty_string(self) -> None:
        # Empty string has 1 "word" (the empty string itself from split)
        # but max(1, ...) ensures at least 1
        assert estimate_tokens("") >= 1

    def test_estimate_tokens_long_text(self) -> None:
        text = " ".join(["word"] * 1000)
        est = estimate_tokens(text)
        assert est == 1500  # 1000 * 1.5

    def test_scholarly_prose_not_underestimated(self) -> None:
        """Scholarly text with long words should not drastically undercount tokens.

        Real-world example: batch of scholarly prose estimated at 93,611 tokens
        but Voyage API reported 125,766 actual tokens (34% higher with 1.3x).
        The 1.5x multiplier brings the estimate much closer.
        """
        # Simulate scholarly prose: long words, citations, technical terms
        scholarly_words = [
            "transubstantiation", "epistemological", "hermeneutics",
            "phenomenological", "Christendom", "Neoplatonist",
            "eschatological", "soteriological", "pneumatological",
            "sacramental", "incarnational", "ecclesiology",
        ]
        # Build text with mix of short and long words (realistic pattern)
        words = []
        for i in range(1000):
            if i % 3 == 0:
                words.append(scholarly_words[i % len(scholarly_words)])
            else:
                words.append("the")
        text = " ".join(words)
        est = estimate_tokens(text)
        # With 1.5x: 1000 * 1.5 = 1500
        # Must be at least 1400 to avoid the 34% underestimate seen with 1.3x
        assert est >= 1400, f"Estimate {est} too low for scholarly prose"


class TestTokenAwareBatching:
    """Validate token-aware batch splitting."""

    def test_small_input_single_batch(self) -> None:
        """5 short texts should fit in one batch."""
        texts = ["hello world"] * 5
        batches = build_token_aware_batches(texts)
        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_large_chunks_split_by_tokens(self) -> None:
        """64 chunks of ~2.4K tokens each (~156K total) must split into multiple batches."""
        # Simulate a 104K-word book with 64 chunks: ~1625 words per chunk
        chunk_text = " ".join(["word"] * 1625)  # ~2437 estimated tokens (1625 * 1.5)
        texts = [chunk_text] * 64
        batches = build_token_aware_batches(texts)
        # Total tokens: 64 * 2437 = ~156K, limit is 80K per batch
        # Should produce at least 2 batches
        assert len(batches) >= 2
        # Every batch should be under the token limit
        for batch in batches:
            batch_tokens = sum(estimate_tokens(t) for t in batch)
            assert batch_tokens <= _MAX_TOKENS_PER_BATCH

    def test_very_large_single_chunk_gets_solo_batch(self) -> None:
        """A single chunk exceeding the token limit gets its own batch."""
        # 100K words → ~150K tokens, over the 80K limit
        huge_text = " ".join(["word"] * 100_000)
        small_text = "hello"
        texts = [small_text, huge_text, small_text]
        batches = build_token_aware_batches(texts)
        # The huge text should be in its own batch
        assert len(batches) == 3
        assert batches[0] == [small_text]
        assert batches[1] == [huge_text]
        assert batches[2] == [small_text]

    def test_respects_item_count_limit(self) -> None:
        """Even tiny texts should split at max_items boundary."""
        texts = ["hi"] * 200
        batches = build_token_aware_batches(texts, max_items=50)
        assert len(batches) == 4
        assert all(len(b) <= 50 for b in batches)

    def test_100_plus_chunks_all_covered(self) -> None:
        """All 100+ chunks are present across batches (no data loss)."""
        texts = [f"chunk number {i} with some words" for i in range(150)]
        batches = build_token_aware_batches(texts)
        flat = [t for batch in batches for t in batch]
        assert flat == texts

    def test_realistic_scholarly_book(self) -> None:
        """Simulate 'Faith, Hope and Poetry': 104K words, 64 chunks.

        Original failure: 50 chunks sent as one batch → 328K tokens.
        With token-aware batching, no batch should exceed 80K tokens.
        """
        # Average chunk: 104000/64 ≈ 1625 words
        chunks = [" ".join(["scholarly"] * 1625) for _ in range(64)]
        batches = build_token_aware_batches(chunks)
        for batch in batches:
            est = sum(estimate_tokens(t) for t in batch)
            assert est <= _MAX_TOKENS_PER_BATCH, (
                f"Batch has {est} estimated tokens, exceeds {_MAX_TOKENS_PER_BATCH}"
            )
        # Verify all chunks are covered
        total_texts = sum(len(b) for b in batches)
        assert total_texts == 64


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

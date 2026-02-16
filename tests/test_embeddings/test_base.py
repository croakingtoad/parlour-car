"""Tests for the embedding provider abstract interface and result types."""

from __future__ import annotations

import pytest

from author_library.embeddings.base import (
    BatchEmbeddingResult,
    EmbeddingProvider,
    EmbeddingResult,
)


class TestEmbeddingResult:
    """Tests for the EmbeddingResult dataclass."""

    def test_creation_minimal(self) -> None:
        result = EmbeddingResult(
            vector=[0.1, 0.2, 0.3],
            model="test-model",
            provider="test",
            dimensions=3,
        )
        assert result.vector == [0.1, 0.2, 0.3]
        assert result.model == "test-model"
        assert result.provider == "test"
        assert result.dimensions == 3
        assert result.token_count is None

    def test_creation_with_token_count(self) -> None:
        result = EmbeddingResult(
            vector=[0.1],
            model="m",
            provider="p",
            dimensions=1,
            token_count=42,
        )
        assert result.token_count == 42

    def test_frozen(self) -> None:
        result = EmbeddingResult(vector=[0.1], model="m", provider="p", dimensions=1)
        with pytest.raises(AttributeError):
            result.model = "other"  # type: ignore[misc]


class TestBatchEmbeddingResult:
    """Tests for the BatchEmbeddingResult dataclass."""

    def test_creation(self) -> None:
        result = BatchEmbeddingResult(
            vectors=[[0.1, 0.2], [0.3, 0.4]],
            model="m",
            provider="p",
            dimensions=2,
        )
        assert len(result.vectors) == 2
        assert result.token_counts is None

    def test_with_token_counts(self) -> None:
        result = BatchEmbeddingResult(
            vectors=[[0.1]],
            model="m",
            provider="p",
            dimensions=1,
            token_counts=[10],
        )
        assert result.token_counts == [10]

    def test_frozen(self) -> None:
        result = BatchEmbeddingResult(vectors=[], model="m", provider="p", dimensions=1)
        with pytest.raises(AttributeError):
            result.model = "other"  # type: ignore[misc]


class TestEmbeddingProviderABC:
    """Tests that EmbeddingProvider cannot be instantiated directly."""

    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            EmbeddingProvider()  # type: ignore[abstract]

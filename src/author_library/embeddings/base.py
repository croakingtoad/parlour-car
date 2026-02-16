"""Abstract embedding provider interface and result types.

Defines the contract that all embedding providers must implement,
plus immutable result dataclasses for single and batch operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingResult:
    """Result of a single text embedding operation."""

    vector: list[float]
    model: str
    provider: str
    dimensions: int
    token_count: int | None = None


@dataclass(frozen=True)
class BatchEmbeddingResult:
    """Result of a batch embedding operation."""

    vectors: list[list[float]]
    model: str
    provider: str
    dimensions: int
    token_counts: list[int | None] | None = None


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers.

    Every provider must implement single-text and batch embedding.
    Providers that distinguish document vs. query embeddings (e.g. Voyage AI)
    should override ``embed_query`` separately.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique provider identifier (e.g., 'voyage', 'openai', 'ollama')."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Current model being used."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Output vector dimensions."""
        ...

    @abstractmethod
    async def embed_text(self, text: str) -> EmbeddingResult:
        """Generate embedding for a single text (document input type)."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        """Generate embeddings for a batch of texts (document input type)."""
        ...

    async def embed_query(self, text: str) -> EmbeddingResult:
        """Generate embedding optimised for search queries.

        Providers that support distinct query embeddings (e.g. Voyage AI)
        should override this.  Default delegates to ``embed_text``.
        """
        return await self.embed_text(text)

    async def close(self) -> None:  # noqa: B027
        """Clean up resources (HTTP clients, etc.)."""

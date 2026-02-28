"""Abstract embedding provider interface and result types.

Defines the contract that all embedding providers must implement,
plus immutable result dataclasses for single and batch operations,
and shared utilities for token-aware batching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

# -- Token estimation and batching utilities ----------------------------------

_DEFAULT_MAX_TOKENS_PER_BATCH = 80_000  # safe headroom below Voyage's 120K limit
_DEFAULT_MAX_ITEMS_PER_BATCH = 128
_TOKENS_PER_WORD_ESTIMATE = 1.5  # scholarly prose with long words/citations needs higher ratio


def estimate_tokens(text: str) -> int:
    """Estimate token count from text using a word-count heuristic.

    Subword tokenizers typically produce ~1.3-1.6 tokens per whitespace
    word depending on vocabulary complexity.  Scholarly prose with long
    words, citations, and technical terms sits at the high end; 1.5 is a
    safe middle ground that avoids API overruns.
    """
    word_count = len(text.split())
    return max(1, int(word_count * _TOKENS_PER_WORD_ESTIMATE))


def build_token_aware_batches(
    texts: list[str],
    max_tokens: int = _DEFAULT_MAX_TOKENS_PER_BATCH,
    max_items: int = _DEFAULT_MAX_ITEMS_PER_BATCH,
) -> list[list[str]]:
    """Split texts into sub-batches respecting both token and item limits.

    Accumulates texts into the current batch until adding the next text
    would exceed either the token budget or the item count limit, then
    starts a new batch.  A single text that exceeds *max_tokens* on its
    own is placed in a solo batch (the API will reject it if truly too
    large, but we don't silently drop content).
    """
    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_tokens = 0

    for text in texts:
        est = estimate_tokens(text)
        if current_batch and (
            current_tokens + est > max_tokens or len(current_batch) >= max_items
        ):
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(text)
        current_tokens += est

    if current_batch:
        batches.append(current_batch)

    return batches


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

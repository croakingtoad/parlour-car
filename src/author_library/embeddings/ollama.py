"""Ollama (local) embedding provider.

Implements the EmbeddingProvider interface using the Ollama REST API
for local model inference.  No API key required.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog

from author_library.errors import EmbeddingError

from .base import BatchEmbeddingResult, EmbeddingProvider, EmbeddingResult

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = "nomic-embed-text"
_DEFAULT_DIMENSIONS = 768


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Ollama local embedding provider.

    Args:
        model: Model name (default ``nomic-embed-text``).
        dimensions: Output dimensions (default 768).
        base_url: Ollama server base URL (default ``http://localhost:11434``).
        client: Optional pre-configured ``httpx.AsyncClient``.
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        dimensions: int = _DEFAULT_DIMENSIONS,
        base_url: str = _DEFAULT_BASE_URL,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=120.0)

    # -- ABC properties -------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    # -- public API -----------------------------------------------------------

    async def embed_text(self, text: str) -> EmbeddingResult:
        vector = await self._embed_single(text)
        return EmbeddingResult(
            vector=vector,
            model=self._model,
            provider=self.provider_name,
            dimensions=self._dimensions,
        )

    async def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        """Embed a batch of texts sequentially (Ollama processes one at a time)."""
        if not texts:
            raise EmbeddingError(
                "Cannot embed empty batch",
                context={"provider": "ollama"},
            )

        vectors: list[list[float]] = []
        for text in texts:
            vector = await self._embed_single(text)
            vectors.append(vector)

        return BatchEmbeddingResult(
            vectors=vectors,
            model=self._model,
            provider=self.provider_name,
            dimensions=self._dimensions,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- internals ------------------------------------------------------------

    async def _embed_single(self, text: str) -> list[float]:
        """Send a single embedding request to Ollama."""
        url = f"{self._base_url}/api/embed"
        payload: dict[str, Any] = {
            "model": self._model,
            "input": text,
        }

        try:
            response = await self._client.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise EmbeddingError(
                f"Cannot connect to Ollama at {self._base_url}. Is Ollama running?",
                context={"provider": "ollama", "base_url": self._base_url},
                cause=exc,
            ) from exc
        except httpx.TimeoutException as exc:
            raise EmbeddingError(
                f"Ollama request timed out ({self._base_url})",
                context={"provider": "ollama", "base_url": self._base_url},
                cause=exc,
            ) from exc

        if response.status_code != 200:
            body = response.text
            raise EmbeddingError(
                f"Ollama API error {response.status_code}: {body}",
                context={
                    "provider": "ollama",
                    "status_code": response.status_code,
                    "model": self._model,
                },
            )

        data = response.json()
        # Ollama /api/embed returns {"embeddings": [[...]]} for single input
        embeddings = data.get("embeddings")
        if not embeddings or not embeddings[0]:
            raise EmbeddingError(
                "Ollama returned empty embeddings",
                context={"provider": "ollama", "model": self._model, "response": data},
            )
        return embeddings[0]  # type: ignore[no-any-return]

"""Voyage AI embedding provider.

Implements the EmbeddingProvider interface using the Voyage AI REST API.
Supports document/query input types and batch embedding with automatic
chunking and retry logic.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from author_library.errors import EmbeddingError

from .base import BatchEmbeddingResult, EmbeddingProvider, EmbeddingResult

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_VOYAGE_API_URL = "https://api.voyageai.com/v1/embeddings"
_DEFAULT_MODEL = "voyage-3-large"
_DEFAULT_DIMENSIONS = 1024
_MAX_BATCH_SIZE = 128
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0


class VoyageEmbeddingProvider(EmbeddingProvider):
    """Voyage AI embedding provider.

    Args:
        api_key: Voyage AI API key.
        model: Model name (default ``voyage-3-large``).
        dimensions: Output dimensions (default 1024).
        client: Optional pre-configured ``httpx.AsyncClient``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        dimensions: int = _DEFAULT_DIMENSIONS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key:
            raise EmbeddingError(
                "Voyage AI API key is required",
                context={"provider": "voyage"},
            )
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=60.0)

    # -- ABC properties -------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "voyage"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    # -- public API -----------------------------------------------------------

    async def embed_text(self, text: str) -> EmbeddingResult:
        """Embed a single text as a document."""
        result = await self._request([text], input_type="document")
        return EmbeddingResult(
            vector=result["data"][0]["embedding"],
            model=result.get("model", self._model),
            provider=self.provider_name,
            dimensions=self._dimensions,
            token_count=result.get("usage", {}).get("total_tokens"),
        )

    async def embed_query(self, text: str) -> EmbeddingResult:
        """Embed a single text as a query (optimised for search)."""
        result = await self._request([text], input_type="query")
        return EmbeddingResult(
            vector=result["data"][0]["embedding"],
            model=result.get("model", self._model),
            provider=self.provider_name,
            dimensions=self._dimensions,
            token_count=result.get("usage", {}).get("total_tokens"),
        )

    async def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        """Embed a batch of texts as documents.

        Voyage AI allows up to ~128 texts per request.  Larger batches are
        automatically split into chunks and results concatenated.
        """
        if not texts:
            raise EmbeddingError(
                "Cannot embed empty batch",
                context={"provider": "voyage"},
            )

        all_vectors: list[list[float]] = []
        all_token_counts: list[int | None] = []

        for i in range(0, len(texts), _MAX_BATCH_SIZE):
            chunk = texts[i : i + _MAX_BATCH_SIZE]
            result = await self._request(chunk, input_type="document")
            # Voyage returns data sorted by index
            sorted_data = sorted(result["data"], key=lambda d: d["index"])
            all_vectors.extend(d["embedding"] for d in sorted_data)
            usage = result.get("usage", {})
            total = usage.get("total_tokens")
            # Voyage doesn't break down per-input tokens; store total for first chunk item
            if total is not None:
                all_token_counts.extend(
                    [total if j == 0 else None for j in range(len(sorted_data))]
                )
            else:
                all_token_counts.extend([None] * len(sorted_data))

        return BatchEmbeddingResult(
            vectors=all_vectors,
            model=self._model,
            provider=self.provider_name,
            dimensions=self._dimensions,
            token_counts=all_token_counts,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- internals ------------------------------------------------------------

    async def _request(self, texts: list[str], *, input_type: str) -> dict[str, Any]:
        """Send an embedding request with retry + exponential backoff."""
        payload: dict[str, Any] = {
            "model": self._model,
            "input": texts,
            "input_type": input_type,
            "output_dimension": self._dimensions,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_exc: BaseException | None = None
        backoff = _INITIAL_BACKOFF

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await self._client.post(
                    _VOYAGE_API_URL,
                    json=payload,
                    headers=headers,
                )

                if response.status_code == 429:
                    retry_after = float(response.headers.get("retry-after", backoff))
                    logger.warning(
                        "voyage_rate_limited",
                        attempt=attempt,
                        retry_after=retry_after,
                    )
                    await asyncio.sleep(retry_after)
                    backoff *= 2
                    continue

                if response.status_code >= 500:
                    logger.warning(
                        "voyage_server_error",
                        status=response.status_code,
                        attempt=attempt,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue

                if response.status_code != 200:
                    body = response.text
                    raise EmbeddingError(
                        f"Voyage API error {response.status_code}: {body}",
                        context={
                            "provider": "voyage",
                            "status_code": response.status_code,
                            "model": self._model,
                        },
                    )

                return response.json()  # type: ignore[no-any-return]

            except httpx.TimeoutException as exc:
                logger.warning("voyage_timeout", attempt=attempt)
                last_exc = exc
                await asyncio.sleep(backoff)
                backoff *= 2
            except httpx.ConnectError as exc:
                logger.warning("voyage_connect_error", attempt=attempt, error=str(exc))
                last_exc = exc
                await asyncio.sleep(backoff)
                backoff *= 2
            except EmbeddingError:
                raise

        raise EmbeddingError(
            f"Voyage API request failed after {_MAX_RETRIES} attempts",
            context={"provider": "voyage", "model": self._model},
            cause=last_exc,
        )

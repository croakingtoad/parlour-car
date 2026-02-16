"""OpenAI embedding provider.

Implements the EmbeddingProvider interface using the OpenAI REST API.
Supports batch embedding with retry logic and exponential backoff.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from author_library.errors import EmbeddingError

from .base import BatchEmbeddingResult, EmbeddingProvider, EmbeddingResult

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_OPENAI_API_URL = "https://api.openai.com/v1/embeddings"
_DEFAULT_MODEL = "text-embedding-3-large"
_DEFAULT_DIMENSIONS = 3072
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 1.0


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider.

    Args:
        api_key: OpenAI API key.
        model: Model name (default ``text-embedding-3-large``).
        dimensions: Output dimensions (default 3072).
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
                "OpenAI API key is required",
                context={"provider": "openai"},
            )
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=60.0)

    # -- ABC properties -------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    # -- public API -----------------------------------------------------------

    async def embed_text(self, text: str) -> EmbeddingResult:
        result = await self._request([text])
        data_item = result["data"][0]
        usage = result.get("usage", {})
        return EmbeddingResult(
            vector=data_item["embedding"],
            model=result.get("model", self._model),
            provider=self.provider_name,
            dimensions=self._dimensions,
            token_count=usage.get("total_tokens"),
        )

    async def embed_batch(self, texts: list[str]) -> BatchEmbeddingResult:
        if not texts:
            raise EmbeddingError(
                "Cannot embed empty batch",
                context={"provider": "openai"},
            )

        result = await self._request(texts)
        sorted_data = sorted(result["data"], key=lambda d: d["index"])
        vectors = [d["embedding"] for d in sorted_data]

        usage = result.get("usage", {})
        total_tokens = usage.get("total_tokens")
        token_counts: list[int | None] | None = None
        if total_tokens is not None:
            token_counts = [total_tokens if i == 0 else None for i in range(len(vectors))]

        return BatchEmbeddingResult(
            vectors=vectors,
            model=result.get("model", self._model),
            provider=self.provider_name,
            dimensions=self._dimensions,
            token_counts=token_counts,
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # -- internals ------------------------------------------------------------

    async def _request(self, texts: list[str]) -> dict[str, Any]:
        """Send an embedding request with retry + exponential backoff."""
        payload: dict[str, Any] = {
            "model": self._model,
            "input": texts,
            "dimensions": self._dimensions,
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
                    _OPENAI_API_URL,
                    json=payload,
                    headers=headers,
                )

                if response.status_code == 429:
                    retry_after = float(response.headers.get("retry-after", backoff))
                    logger.warning(
                        "openai_rate_limited",
                        attempt=attempt,
                        retry_after=retry_after,
                    )
                    await asyncio.sleep(retry_after)
                    backoff *= 2
                    continue

                if response.status_code >= 500:
                    logger.warning(
                        "openai_server_error",
                        status=response.status_code,
                        attempt=attempt,
                    )
                    await asyncio.sleep(backoff)
                    backoff *= 2
                    continue

                if response.status_code != 200:
                    body = response.text
                    raise EmbeddingError(
                        f"OpenAI API error {response.status_code}: {body}",
                        context={
                            "provider": "openai",
                            "status_code": response.status_code,
                            "model": self._model,
                        },
                    )

                return response.json()  # type: ignore[no-any-return]

            except httpx.TimeoutException as exc:
                logger.warning("openai_timeout", attempt=attempt)
                last_exc = exc
                await asyncio.sleep(backoff)
                backoff *= 2
            except httpx.ConnectError as exc:
                logger.warning("openai_connect_error", attempt=attempt, error=str(exc))
                last_exc = exc
                await asyncio.sleep(backoff)
                backoff *= 2
            except EmbeddingError:
                raise

        raise EmbeddingError(
            f"OpenAI API request failed after {_MAX_RETRIES} attempts",
            context={"provider": "openai", "model": self._model},
            cause=last_exc,
        )

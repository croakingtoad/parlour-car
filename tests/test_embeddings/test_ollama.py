"""Tests for the Ollama embedding provider.

Integration tests skip if the local Ollama server is not reachable.
"""

from __future__ import annotations

import httpx
import pytest

from author_library.embeddings.ollama import (
    _DEFAULT_BASE_URL,
    OllamaEmbeddingProvider,
)
from author_library.errors import EmbeddingError


def _ollama_reachable() -> bool:
    """Check if a local Ollama server is available."""
    try:
        resp = httpx.get(f"{_DEFAULT_BASE_URL}/api/tags", timeout=2.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


class TestOllamaRequestConstruction:
    """Validate that Ollama provider is configured correctly."""

    def test_provider_properties(self) -> None:
        provider = OllamaEmbeddingProvider()
        assert provider.provider_name == "ollama"
        assert provider.model_name == "nomic-embed-text"
        assert provider.dimensions == 768

    def test_custom_config(self) -> None:
        provider = OllamaEmbeddingProvider(
            model="all-minilm",
            dimensions=384,
            base_url="http://gpu-box:11434",
        )
        assert provider.model_name == "all-minilm"
        assert provider.dimensions == 384

    def test_base_url_trailing_slash_stripped(self) -> None:
        provider = OllamaEmbeddingProvider(base_url="http://localhost:11434/")
        assert provider._base_url == "http://localhost:11434"


class TestOllamaEmptyBatch:
    """Validate error on empty batch."""

    @pytest.mark.asyncio
    async def test_empty_batch_raises(self) -> None:
        provider = OllamaEmbeddingProvider()
        with pytest.raises(EmbeddingError, match="empty batch"):
            await provider.embed_batch([])
        await provider.close()


class TestOllamaConnectionError:
    """Validate graceful handling when Ollama is not running."""

    @pytest.mark.asyncio
    async def test_connection_refused(self) -> None:
        provider = OllamaEmbeddingProvider(base_url="http://localhost:1")
        with pytest.raises(EmbeddingError, match="Cannot connect to Ollama"):
            await provider.embed_text("test")
        await provider.close()


# -- Integration tests (require local Ollama) --------------------------------

_has_ollama = _ollama_reachable()


@pytest.mark.skipif(not _has_ollama, reason="Ollama not reachable at localhost:11434")
class TestOllamaIntegration:
    """Integration tests against a running Ollama instance."""

    @pytest.fixture
    def provider(self) -> OllamaEmbeddingProvider:
        return OllamaEmbeddingProvider()

    @pytest.mark.asyncio
    async def test_embed_text(self, provider: OllamaEmbeddingProvider) -> None:
        result = await provider.embed_text("Hello, world!")
        assert len(result.vector) > 0
        assert result.provider == "ollama"
        await provider.close()

    @pytest.mark.asyncio
    async def test_embed_batch(self, provider: OllamaEmbeddingProvider) -> None:
        texts = ["First text", "Second text"]
        result = await provider.embed_batch(texts)
        assert len(result.vectors) == 2
        await provider.close()

"""Tests for the Ollama embedding provider.

Integration tests skip unless the local Ollama server has the required model.
"""

from __future__ import annotations

import httpx
import pytest

from author_library.embeddings.ollama import (
    _DEFAULT_BASE_URL,
    _DEFAULT_MODEL,
    OllamaEmbeddingProvider,
)
from author_library.errors import EmbeddingError


def _ollama_model_available() -> bool:
    """Check whether the local Ollama server has the integration-test model."""
    try:
        resp = httpx.get(f"{_DEFAULT_BASE_URL}/api/tags", timeout=2.0)
        resp.raise_for_status()
        models = resp.json().get("models", [])
        return any(model.get("name", "").partition(":")[0] == _DEFAULT_MODEL for model in models)
    except (httpx.HTTPError, TypeError, ValueError):
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

_has_ollama_model = _ollama_model_available()


@pytest.mark.skipif(
    not _has_ollama_model,
    reason=f"Ollama is unavailable or {_DEFAULT_MODEL} is not installed",
)
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

"""Tests for the embedding provider registry."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from author_library.config import APIKeySettings, EmbeddingSettings, Settings
from author_library.embeddings import ProviderRegistry
from author_library.embeddings.ollama import OllamaEmbeddingProvider
from author_library.embeddings.openai import OpenAIEmbeddingProvider
from author_library.embeddings.voyage import VoyageEmbeddingProvider
from author_library.errors import ConfigurationError, EmbeddingError


class TestProviderRegistration:
    """Tests for provider registration mechanics."""

    def test_builtin_providers_registered(self) -> None:
        available = ProviderRegistry.available_providers()
        assert "voyage" in available
        assert "openai" in available
        assert "ollama" in available

    def test_available_providers_sorted(self) -> None:
        available = ProviderRegistry.available_providers()
        assert available == sorted(available)


class TestProviderCreation:
    """Tests for creating providers from settings."""

    def test_create_voyage_provider(self) -> None:
        settings = Settings(
            embedding=EmbeddingSettings(
                provider="voyage",
                model="voyage-3-large",
                dimensions=1024,
            ),
            api_keys=APIKeySettings(
                voyage_api_key=SecretStr("test-voyage-key"),
            ),
        )
        provider = ProviderRegistry.create(settings)
        assert isinstance(provider, VoyageEmbeddingProvider)
        assert provider.provider_name == "voyage"
        assert provider.model_name == "voyage-3-large"
        assert provider.dimensions == 1024

    def test_create_openai_provider(self) -> None:
        settings = Settings(
            embedding=EmbeddingSettings(
                provider="openai",
                model="text-embedding-3-large",
                dimensions=3072,
            ),
            api_keys=APIKeySettings(
                openai_api_key=SecretStr("test-openai-key"),
            ),
        )
        provider = ProviderRegistry.create(settings)
        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.provider_name == "openai"
        assert provider.model_name == "text-embedding-3-large"
        assert provider.dimensions == 3072

    def test_create_ollama_provider(self) -> None:
        settings = Settings(
            embedding=EmbeddingSettings(
                provider="ollama",
                model="nomic-embed-text",
                dimensions=768,
            ),
        )
        provider = ProviderRegistry.create(settings)
        assert isinstance(provider, OllamaEmbeddingProvider)
        assert provider.provider_name == "ollama"
        assert provider.model_name == "nomic-embed-text"
        assert provider.dimensions == 768

    def test_unknown_provider_raises(self) -> None:
        settings = Settings(
            embedding=EmbeddingSettings(provider="nonexistent"),
        )
        with pytest.raises(ConfigurationError, match="Unknown embedding provider"):
            ProviderRegistry.create(settings)

    def test_voyage_missing_key_raises(self) -> None:
        settings = Settings(
            embedding=EmbeddingSettings(provider="voyage"),
            api_keys=APIKeySettings(voyage_api_key=None),
        )
        with pytest.raises(EmbeddingError, match="VOYAGE_API_KEY"):
            ProviderRegistry.create(settings)

    def test_openai_missing_key_raises(self) -> None:
        settings = Settings(
            embedding=EmbeddingSettings(provider="openai"),
            api_keys=APIKeySettings(openai_api_key=None),
        )
        with pytest.raises(EmbeddingError, match="OPENAI_API_KEY"):
            ProviderRegistry.create(settings)

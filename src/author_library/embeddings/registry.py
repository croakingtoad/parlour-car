"""Embedding provider registry.

Creates provider instances from application configuration and maintains
a registry of available provider classes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import structlog

from author_library.errors import ConfigurationError, EmbeddingError

if TYPE_CHECKING:
    from author_library.config import Settings

    from .base import EmbeddingProvider

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class ProviderRegistry:
    """Registry of available embedding providers.

    Creates provider instances based on configuration.
    Validates that required API keys are present before instantiation.
    """

    _providers: ClassVar[dict[str, type[EmbeddingProvider]]] = {}

    @classmethod
    def register(cls, name: str, provider_cls: type[EmbeddingProvider]) -> None:
        """Register a provider class under the given name."""
        cls._providers[name] = provider_cls
        logger.debug("embedding_provider_registered", name=name)

    @classmethod
    def create(cls, settings: Settings) -> EmbeddingProvider:
        """Create a provider instance from application settings.

        Reads ``settings.embedding.provider`` to select the class,
        then validates that any required API key is present.

        Raises:
            ConfigurationError: If the provider name is unknown.
            EmbeddingError: If a required API key is missing.
        """
        provider_name = settings.embedding.provider
        provider_cls = cls._providers.get(provider_name)
        if provider_cls is None:
            raise ConfigurationError(
                f"Unknown embedding provider: {provider_name!r}. "
                f"Available: {', '.join(sorted(cls._providers))}",
                context={"provider": provider_name},
            )

        kwargs: dict[str, Any] = {
            "model": settings.embedding.model,
            "dimensions": settings.embedding.dimensions,
        }

        if provider_name == "voyage":
            api_key_secret = settings.api_keys.voyage_api_key
            if api_key_secret is None:
                raise EmbeddingError(
                    "Voyage AI requires VOYAGE_API_KEY to be set",
                    context={"provider": "voyage"},
                )
            kwargs["api_key"] = api_key_secret.get_secret_value()

        elif provider_name == "openai":
            api_key_secret = settings.api_keys.openai_api_key
            if api_key_secret is None:
                raise EmbeddingError(
                    "OpenAI requires OPENAI_API_KEY to be set",
                    context={"provider": "openai"},
                )
            kwargs["api_key"] = api_key_secret.get_secret_value()

        # ollama needs no API key

        logger.info(
            "embedding_provider_created",
            provider=provider_name,
            model=settings.embedding.model,
            dimensions=settings.embedding.dimensions,
        )
        return provider_cls(**kwargs)

    @classmethod
    def available_providers(cls) -> list[str]:
        """Return sorted list of registered provider names."""
        return sorted(cls._providers)

    @classmethod
    def _clear(cls) -> None:
        """Remove all registered providers (for testing only)."""
        cls._providers.clear()

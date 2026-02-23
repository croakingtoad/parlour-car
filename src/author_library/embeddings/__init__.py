"""Embedding provider abstraction layer.

Exports the abstract interface, result types, concrete providers,
and the provider registry.  Built-in providers are auto-registered
on import.
"""

from .base import BatchEmbeddingResult, EmbeddingProvider, EmbeddingResult
from .cached import CachedEmbeddingProvider
from .ollama import OllamaEmbeddingProvider
from .openai import OpenAIEmbeddingProvider
from .registry import ProviderRegistry
from .voyage import VoyageEmbeddingProvider

# Auto-register built-in providers
ProviderRegistry.register("voyage", VoyageEmbeddingProvider)
ProviderRegistry.register("openai", OpenAIEmbeddingProvider)
ProviderRegistry.register("ollama", OllamaEmbeddingProvider)

__all__ = [
    "BatchEmbeddingResult",
    "CachedEmbeddingProvider",
    "EmbeddingProvider",
    "EmbeddingResult",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "ProviderRegistry",
    "VoyageEmbeddingProvider",
]

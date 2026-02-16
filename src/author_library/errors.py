"""Error hierarchy for The Author Library.

All errors include a context dict for structured logging and an optional
cause for exception chaining.
"""

from __future__ import annotations

from typing import Any


class AuthorLibraryError(Exception):
    """Base exception for all Author Library errors."""

    def __init__(
        self,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.context: dict[str, Any] = context or {}
        if cause is not None:
            self.__cause__ = cause


class IngestionError(AuthorLibraryError):
    """Raised when document ingestion fails."""


class ClassificationError(AuthorLibraryError):
    """Raised when source classification fails."""


class RetrievalError(AuthorLibraryError):
    """Raised when retrieval operations fail."""


class EmbeddingError(AuthorLibraryError):
    """Raised when embedding generation or lookup fails."""


class StorageError(AuthorLibraryError):
    """Raised when database operations fail."""


class ConfigurationError(AuthorLibraryError):
    """Raised when configuration is invalid or missing."""


class ParsingError(AuthorLibraryError):
    """Raised when document parsing fails."""

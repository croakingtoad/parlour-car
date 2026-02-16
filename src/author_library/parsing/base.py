"""Abstract base class for document parsers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from author_library.parsing.models import ParsedDocument


class DocumentParser(ABC):
    """Base class for all document parsers.

    Every format-specific parser must subclass this and implement
    ``parse`` and ``supported_extensions``.
    """

    @abstractmethod
    async def parse(self, source: Path | str) -> ParsedDocument:
        """Parse a document and return structured output.

        Args:
            source: Path to the document file.

        Returns:
            A fully-populated ParsedDocument.

        Raises:
            ParsingError: When the document cannot be parsed.
        """
        ...

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of file extensions this parser handles (including dot)."""
        ...

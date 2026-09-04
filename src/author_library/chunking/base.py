"""Abstract base class for genre-specific chunking strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from author_library.chunking.models import Chunk
    from author_library.parsing.models import ParsedDocument


class ChunkingStrategy(ABC):
    """Base class for genre-specific chunking strategies.

    Each strategy receives a ParsedDocument (the document tree produced by
    the parsing layer) and emits a flat list of Chunks at all three
    granularity tiers.  The strategy decides how to walk the tree and where
    to place boundaries based on genre conventions.
    """

    @abstractmethod
    def chunk(
        self,
        document: ParsedDocument,
        work_id: str,
        source_class: str,
    ) -> list[Chunk]:
        """Chunk a parsed document into multi-granularity chunks.

        Args:
            document: The parsed document tree from the parsing layer.
            work_id: Catalog identifier for the work (e.g. ``guite--faith-hope-poetry``).
            source_class: Source classification
                (primary, secondary, contextual, tertiary, personal, reference).

        Returns:
            A flat list of Chunk objects at macro, meso, and micro granularity.
        """
        ...

    @abstractmethod
    def supported_genres(self) -> list[str]:
        """Return genre tags this strategy handles.

        The registry matches a work's ``genre_tags`` against these to select
        the appropriate strategy.
        """
        ...

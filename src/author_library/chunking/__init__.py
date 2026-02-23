"""Genre-aware chunking subsystem — strategy registry and public API.

Usage::

    from author_library.chunking import get_chunking_strategy

    strategy = get_chunking_strategy(["scholarly_prose", "theology"])
    chunks = strategy.chunk(parsed_doc, work_id="guite--faith-hope-poetry", source_class="primary")
"""

from __future__ import annotations

from author_library.chunking.base import ChunkingStrategy
from author_library.chunking.correspondence import (
    BlogStrategy,
    InterviewStrategy,
    LetterStrategy,
)
from author_library.chunking.models import Chunk, ChunkGranularity
from author_library.chunking.poetry import PoetryStrategy
from author_library.chunking.scholarly import ScholarlyProseStrategy
from author_library.chunking.sermon import SermonStrategy
from author_library.chunking.transcript import TranscriptChunkingStrategy
from author_library.errors import IngestionError

# All registered strategies, in priority order.
# The first strategy whose supported_genres() intersects with the
# work's genre_tags wins.
_STRATEGIES: list[ChunkingStrategy] = [
    PoetryStrategy(),
    TranscriptChunkingStrategy(),  # before interview (more specific)
    InterviewStrategy(),
    LetterStrategy(),
    BlogStrategy(),
    SermonStrategy(),
    ScholarlyProseStrategy(),  # broadest match — last
]

# Build a lookup: genre tag → strategy
_GENRE_MAP: dict[str, ChunkingStrategy] = {}
for _strategy in _STRATEGIES:
    for _genre in _strategy.supported_genres():
        _GENRE_MAP[_genre.lower()] = _strategy


def get_chunking_strategy(genre_tags: list[str]) -> ChunkingStrategy:
    """Return the appropriate chunking strategy for the given genre tags.

    Matches against registered strategies in priority order.  If no genre
    tag matches, falls back to the scholarly prose strategy (broadest).

    Args:
        genre_tags: Genre tags from the work's catalog entry.

    Returns:
        A ChunkingStrategy instance appropriate for the genres.

    Raises:
        IngestionError: If genre_tags is empty.
    """
    if not genre_tags:
        raise IngestionError(
            "Cannot select chunking strategy: no genre tags provided",
            context={"genre_tags": genre_tags},
        )

    for tag in genre_tags:
        strategy = _GENRE_MAP.get(tag.lower())
        if strategy is not None:
            return strategy

    # Default fallback: scholarly prose (most general)
    return ScholarlyProseStrategy()


def list_strategies() -> list[ChunkingStrategy]:
    """Return all registered chunking strategies."""
    return list(_STRATEGIES)


__all__ = [
    "BlogStrategy",
    "Chunk",
    "ChunkGranularity",
    "ChunkingStrategy",
    "InterviewStrategy",
    "LetterStrategy",
    "PoetryStrategy",
    "ScholarlyProseStrategy",
    "SermonStrategy",
    "TranscriptChunkingStrategy",
    "get_chunking_strategy",
    "list_strategies",
]

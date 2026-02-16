"""Chunk data model and granularity enum for multi-level document chunking."""

from __future__ import annotations

import uuid
from enum import StrEnum

from pydantic import BaseModel, Field


class ChunkGranularity(StrEnum):
    """Three-tier granularity for document chunks."""

    MACRO = "macro"  # 500-1500 words: chapter summaries, high-level overviews
    MESO = "meso"  # 150-500 words: section/argument-level, primary retrieval unit
    MICRO = "micro"  # 30-200 words: paragraph/passage-level, fine-grained matching


# Word count boundaries per granularity tier.
GRANULARITY_BOUNDS: dict[ChunkGranularity, tuple[int, int]] = {
    ChunkGranularity.MACRO: (500, 1500),
    ChunkGranularity.MESO: (150, 500),
    ChunkGranularity.MICRO: (30, 200),
}


class Chunk(BaseModel):
    """A single chunk produced by a genre-aware chunking strategy.

    Every chunk carries its source classification, structural position, and
    optional contextual annotation (prepended before embedding).
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    text: str
    annotation: str | None = None
    granularity: ChunkGranularity
    work_id: str
    source_class: str  # primary, secondary, contextual, tertiary
    chapter: str | None = None
    section: str | None = None
    position: int  # ordering within the work at this granularity
    parent_chunk_id: str | None = None  # parent in granularity hierarchy
    metadata: dict[str, str | int | bool | list[str]] = Field(default_factory=dict)

    @property
    def word_count(self) -> int:
        """Number of whitespace-delimited words in the chunk text."""
        return len(self.text.split())

    @property
    def annotated_text(self) -> str:
        """Text with contextual annotation prepended (for embedding)."""
        if self.annotation:
            return f"{self.annotation}\n\n{self.text}"
        return self.text
